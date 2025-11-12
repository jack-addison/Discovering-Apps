#!/usr/bin/env python3
"""Generate embeddings for dissatisfied apps stored in Neon."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

import numpy as np
import psycopg
from openai import OpenAI

from ..config import load_settings

DEFAULT_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed dissatisfied app descriptions in Neon.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI embedding model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        action="append",
        help="Restrict to specific run IDs (repeatable).",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        help="Optional max number of apps to embed (for smoke tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of descriptions per OpenAI call (default: 50).",
    )
    parser.add_argument(
        "--postgres-dsn",
        help="Override PROTOTYPE_DATABASE_URL for Neon.",
    )
    parser.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Embed every snapshot (instead of only dissatisfied apps).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def fetch_candidates(
    conn: psycopg.Connection,
    *,
    model: str,
    run_ids: Sequence[int] | None,
    limit: int | None,
    embed_all: bool,
) -> List[tuple[int, int, str]]:
    params: List[object] = [model]
    if run_ids:
        placeholders = ",".join(["%s"] * len(run_ids))
        run_filter = f"AND s.run_id IN ({placeholders})"
        params.extend(run_ids)
    else:
        run_filter = ""
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    dissatisfied_join = """
        FROM app_snapshot_dissatisfied d
        JOIN app_snapshots s ON d.run_id = s.run_id AND d.track_id = s.track_id
    """
    all_join = "FROM app_snapshots s"

    def _execute(sql: str) -> List[tuple[int, int, str]]:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    if embed_all:
        query = f"""
            SELECT s.run_id, s.track_id, s.description
            {all_join}
            LEFT JOIN app_snapshot_embeddings e
              ON e.run_id = s.run_id AND e.track_id = s.track_id AND e.model = %s
            WHERE s.description IS NOT NULL
              AND e.run_id IS NULL
              {run_filter}
            ORDER BY s.run_id DESC, s.track_id
            {limit_clause}
        """
        return _execute(query)

    query = f"""
        SELECT d.run_id, d.track_id, s.description
        {dissatisfied_join}
        LEFT JOIN app_snapshot_embeddings e
          ON e.run_id = d.run_id AND e.track_id = d.track_id AND e.model = %s
        WHERE s.description IS NOT NULL
          AND e.run_id IS NULL
          {run_filter}
        ORDER BY d.run_id DESC, d.track_id
        {limit_clause}
    """
    return _execute(query)


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def embed_batch(client: OpenAI, model: str, texts: Sequence[str]) -> List[List[float]]:
    response = client.embeddings.create(model=model, input=list(texts))
    return [record.embedding for record in response.data]


def upsert_embeddings(
    conn: psycopg.Connection,
    *,
    rows: Sequence[tuple[int, int, str]],
    embeddings: Sequence[Sequence[float]],
    model: str,
) -> None:
    timestamp = datetime.now(timezone.utc)
    payload = []
    for (run_id, track_id, description), vector in zip(rows, embeddings, strict=True):
        if not description:
            continue
        digest = hashlib.sha256(description.encode("utf-8")).hexdigest()
        payload.append(
            (
                run_id,
                track_id,
                model,
                digest,
                json.dumps(vector),
                len(vector),
                timestamp,
            )
        )
    if not payload:
        return
    insert_sql = """
        INSERT INTO app_snapshot_embeddings (
            run_id, track_id, model, description_sha256, embedding_json, vector_length, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, track_id, model) DO UPDATE SET
            description_sha256 = EXCLUDED.description_sha256,
            embedding_json = EXCLUDED.embedding_json,
            vector_length = EXCLUDED.vector_length,
            created_at = EXCLUDED.created_at
    """
    with conn.cursor() as cur:
        cur.executemany(insert_sql, payload)
    conn.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn
    client = OpenAI()

    with psycopg.connect(dsn) as conn:
        run_ids = sorted(set(args.run_id)) if args.run_id else None
        fetch_candidates.embed_all = args.all_snapshots
        candidates = fetch_candidates(
            conn,
            model=args.model,
            run_ids=run_ids,
            limit=args.max_apps,
            embed_all=args.all_snapshots,
        )
        logging.info("Found %s embeddings to generate", len(candidates))
        if not candidates:
            return

        for batch in chunked(candidates, args.batch_size):
            texts = [row[2] for row in batch]
            vectors = embed_batch(client, args.model, texts)
            upsert_embeddings(conn, rows=batch, embeddings=vectors, model=args.model)
        logging.info("Embedding generation complete")


if __name__ == "__main__":
    main()
