#!/usr/bin/env python3
"""Cloud variant of embedding generation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import sqlitecloud
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.generate_embeddings import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_WAIT,
    SnapshotRecord,
    ensure_embedding_table,
    compute_hash,
    build_embedding_payload,
    chunked,
    embed_batch,
    upsert_embeddings,
)
from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings directly in SQLiteCloud.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-wait", type=int, default=DEFAULT_RETRY_WAIT)
    parser.add_argument("--max-apps", type=int)
    parser.add_argument("--run-id", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Environment variable OPENAI_API_KEY must be set.")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with sqlitecloud.connect(args.connection_uri) as conn:
        ensure_embedding_table(conn)
        snapshots = fetch_snapshots_cloud(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        to_process = []
        skipped = 0
        for record in snapshots:
            desc_hash = compute_hash(record.description)
            if not args.force and record.existing_hash == desc_hash:
                skipped += 1
                continue
            to_process.append((record, desc_hash))
        logging.info("Snapshots to embed: %d (skipped %d unchanged)", len(to_process), skipped)
        if not to_process:
            return

        client = OpenAI()
        total_batches = (len(to_process) + args.batch_size - 1) // args.batch_size
        processed = 0
        for batch_index, batch in enumerate(chunked(to_process, args.batch_size), start=1):
            payloads = [build_embedding_payload(record) for record, _ in batch]
            embeddings = embed_batch(
                client,
                args.model,
                payloads,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            upsert_embeddings(conn, batch, embeddings, args.model)
            processed += len(batch)
            logging.info("Processed batch %d/%d (%d/%d rows)", batch_index, total_batches, processed, len(to_process))


def fetch_snapshots_cloud(
    connection,
    *,
    model: str,
    run_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[SnapshotRecord]:
    filters = ["TRIM(COALESCE(s.description, '')) <> ''"]
    params: List[Any] = [model]
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"s.run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = "WHERE " + " AND ".join(filters)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT
            s.run_id,
            s.track_id,
            s.name,
            s.primary_genre_name,
            s.price,
            s.currency,
            s.is_free,
            s.success_score,
            s.build_time_estimate,
            s.user_rating_count,
            s.description,
            e.description_sha256
        FROM app_snapshots AS s
        LEFT JOIN app_snapshot_embeddings AS e
            ON e.run_id = s.run_id
           AND e.track_id = s.track_id
           AND e.model = ?
        {where_clause}
        ORDER BY s.run_id DESC, s.track_id
        {limit_clause}
    """
    cursor = connection.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    snapshots: List[SnapshotRecord] = []
    for row in rows:
        record = dict(zip(columns, row))
        snapshots.append(
            SnapshotRecord(
                run_id=record["run_id"],
                track_id=record["track_id"],
                name=record["name"],
                category=record["primary_genre_name"],
                price=record["price"],
                currency=record["currency"],
                is_free=bool(record["is_free"]),
                success_score=record["success_score"],
                build_time_estimate=record["build_time_estimate"],
                number_of_ratings=record["user_rating_count"],
                description=record["description"] or "",
                existing_hash=record["description_sha256"],
            )
        )
    return snapshots


if __name__ == "__main__":
    main()
