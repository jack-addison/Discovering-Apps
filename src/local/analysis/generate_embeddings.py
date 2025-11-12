#!/usr/bin/env python3
"""Generate text embeddings for App Store snapshots to enable similarity search."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from openai import APIError, OpenAI, RateLimitError

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_WAIT = 5


@dataclass
class SnapshotRecord:
    run_id: int
    track_id: int
    name: str
    category: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    is_free: bool
    success_score: Optional[float]
    build_time_estimate: Optional[float]
    number_of_ratings: Optional[int]
    description: str
    existing_hash: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI embeddings for app snapshots to support similarity analysis."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the snapshot database (default: exports/app_store_apps_v2.db).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of descriptions to embed per request (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries when the OpenAI API rate-limits or errors (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=DEFAULT_RETRY_WAIT,
        help=f"Seconds to wait between retries (default: {DEFAULT_RETRY_WAIT}).",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        help="Optional cap on the number of snapshots to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        action="append",
        help="Limit processing to specific scrape run IDs (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed snapshots even if a matching hash exists for the chosen model.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Environment variable OPENAI_API_KEY must be set before running this script.")

    return args


def ensure_embedding_table(connection: sqlite3.Connection) -> None:
    """Create the embedding table if missing."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_snapshot_embeddings (
            run_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            description_sha256 TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            vector_length INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, track_id, model)
        )
        """
    )
    connection.commit()


def fetch_snapshots(
    connection: sqlite3.Connection,
    *,
    model: str,
    run_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[SnapshotRecord]:
    """Fetch app snapshots with existing hash (if any)."""
    filters: List[str] = ["TRIM(COALESCE(s.description, '')) <> ''"]
    params: List[object] = [model]
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
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, params).fetchall()
    snapshots = [
        SnapshotRecord(
            run_id=row["run_id"],
            track_id=row["track_id"],
            name=row["name"],
            category=row["primary_genre_name"],
            price=row["price"],
            currency=row["currency"],
            is_free=bool(row["is_free"]),
            success_score=row["success_score"],
            build_time_estimate=row["build_time_estimate"],
            number_of_ratings=row["user_rating_count"],
            description=row["description"] or "",
            existing_hash=row["description_sha256"],
        )
        for row in rows
    ]
    return snapshots


def compute_hash(text: str) -> str:
    """Return a stable hash of the description for change detection."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_embedding_payload(record: SnapshotRecord) -> str:
    """Enrich the description with structured metadata for higher-quality vectors."""
    price_label = "Free" if record.is_free or (record.price or 0) == 0 else f"{record.price:.2f} {record.currency or 'USD'}"
    success_label = f"{record.success_score:.1f}" if record.success_score is not None else "unknown"
    build_label = f"{record.build_time_estimate:.1f}" if record.build_time_estimate is not None else "unknown"
    ratings_label = f"{record.number_of_ratings:,}" if record.number_of_ratings is not None else "unknown"
    parts = [
        f"Name: {record.name}",
        f"Category: {record.category or 'Unknown'}",
        f"Price: {price_label}",
        f"Success score: {success_label}",
        f"Build weeks: {build_label}",
        f"Rating count: {ratings_label}",
        "Description:",
        record.description.strip(),
    ]
    return "\n".join(parts)


def chunked(iterable: Sequence[Tuple[SnapshotRecord, str]], size: int) -> Iterable[List[Tuple[SnapshotRecord, str]]]:
    for start in range(0, len(iterable), size):
        yield list(iterable[start : start + size])


def embed_batch(
    client: OpenAI,
    model: str,
    payloads: List[str],
    *,
    max_retries: int,
    retry_wait: int,
) -> List[List[float]]:
    attempt = 0
    while True:
        try:
            response = client.embeddings.create(model=model, input=payloads)
            return [item.embedding for item in response.data]
        except RateLimitError as err:
            attempt += 1
            if attempt > max_retries:
                raise
            logging.warning("Rate limited (attempt %s/%s): %s", attempt, max_retries, err)
            time.sleep(retry_wait)
        except APIError as err:
            attempt += 1
            if attempt > max_retries:
                raise
            logging.warning("OpenAI API error (attempt %s/%s): %s", attempt, max_retries, err)
            time.sleep(retry_wait)


def upsert_embeddings(
    connection: sqlite3.Connection,
    records_with_hash: Sequence[Tuple[SnapshotRecord, str]],
    embeddings: Sequence[List[float]],
    model: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    for (record, desc_hash), vector in zip(records_with_hash, embeddings):
        connection.execute(
            """
            INSERT INTO app_snapshot_embeddings (
                run_id,
                track_id,
                model,
                description_sha256,
                embedding_json,
                vector_length,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, track_id, model) DO UPDATE SET
                description_sha256=excluded.description_sha256,
                embedding_json=excluded.embedding_json,
                vector_length=excluded.vector_length,
                created_at=excluded.created_at
            """,
            (
                record.run_id,
                record.track_id,
                model,
                desc_hash,
                json.dumps(vector),
                len(vector),
                timestamp,
            ),
        )
    connection.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading snapshots from %s", args.db_path)
    with sqlite3.connect(args.db_path) as conn:
        ensure_embedding_table(conn)

        snapshots = fetch_snapshots(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        logging.info("Fetched %d snapshots matching criteria", len(snapshots))

        to_process: List[Tuple[SnapshotRecord, str]] = []
        skipped = 0
        for record in snapshots:
            desc_hash = compute_hash(record.description)
            if not args.force and record.existing_hash == desc_hash:
                skipped += 1
                continue
            to_process.append((record, desc_hash))

        logging.info("Snapshots to embed: %d (skipped %d unchanged)", len(to_process), skipped)
        if not to_process:
            logging.info("All embeddings up to date. Nothing to do.")
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
            logging.info(
                "Processed batch %d/%d (%d/%d snapshots)",
                batch_index,
                total_batches,
                processed,
                len(to_process),
            )

    logging.info("Embedding generation complete.")


if __name__ == "__main__":
    main()
