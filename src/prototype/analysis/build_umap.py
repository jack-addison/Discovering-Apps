#!/usr/bin/env python3
"""Compute UMAP coordinates for app snapshots stored in Neon/PostgreSQL."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import psycopg
import umap

from ..config import load_settings

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_SCOPE = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UMAP projection for Neon embeddings.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id (default: text-embedding-3-small)")
    parser.add_argument("--scope-label", default=DEFAULT_SCOPE, help="Label stored in app_snapshot_umap.scope (default: all)")
    parser.add_argument("--n-neighbors", type=int, default=25, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.15, help="UMAP min_dist parameter")
    parser.add_argument("--max-apps", type=int, help="Optional cap for embeddings to project")
    parser.add_argument("--run-id", action="append", type=int, help="Restrict to specific run IDs (appendable)")
    parser.add_argument("--postgres-dsn", help="Override PROTOTYPE_DATABASE_URL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP")
    return parser.parse_args()


def fetch_embeddings(
    conn: psycopg.Connection,
    *,
    model: str,
    limit: int | None,
    run_ids: Sequence[int] | None,
) -> List[Tuple[int, int, np.ndarray]]:
    query = """
        SELECT e.run_id, e.track_id, e.embedding_json
        FROM app_snapshot_embeddings e
        WHERE e.model = %s
        ORDER BY e.run_id DESC, e.track_id
    """
    params: List[object] = [model]
    if run_ids:
        placeholders = ", ".join(["%s"] * len(run_ids))
        query = query.replace("ORDER BY", f"AND e.run_id IN ({placeholders}) ORDER BY", 1)
        params.extend(run_ids)
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    latest: Dict[int, Tuple[int, int, np.ndarray]] = {}
    for run_id, track_id, embedding_json in rows:
        vector = np.array(json.loads(embedding_json), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        vector /= norm
        existing = latest.get(track_id)
        if existing is None or run_id > existing[0]:
            latest[track_id] = (run_id, track_id, vector)
    return list(latest.values())


def upsert_umap(
    conn: psycopg.Connection,
    *,
    scope: str,
    model: str,
    rows: Sequence[Tuple[int, int, float, float]],
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM app_snapshot_umap WHERE scope = %s AND model = %s",
            (scope, model),
        )
        payload = [
            (run_id, track_id, model, scope, float(x), float(y), timestamp)
            for run_id, track_id, x, y in rows
        ]
        cur.executemany(
            """
            INSERT INTO app_snapshot_umap (
                run_id, track_id, model, scope, umap_x, umap_y, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            payload,
        )
    conn.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn

    with psycopg.connect(dsn) as conn:
        run_ids = sorted(set(args.run_id or []), reverse=True) or None
        embeddings = fetch_embeddings(
            conn,
            model=args.model,
            limit=args.max_apps,
            run_ids=run_ids,
        )
        logging.info("Loaded %s embeddings for model '%s'.", len(embeddings), args.model)
        if len(embeddings) < 10:
            logging.warning("Not enough embeddings to compute UMAP projection.")
            return
        matrix = np.vstack([row[2] for row in embeddings])
        reducer = umap.UMAP(
            n_neighbors=min(args.n_neighbors, len(embeddings) - 1),
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.random_state,
        )
        logging.info("Fitting UMAP on %s embeddings (n_neighbors=%s, min_dist=%s)...", len(embeddings), reducer.n_neighbors, args.min_dist)
        start = time.time()
        coords = reducer.fit_transform(matrix)
        logging.info("UMAP fit completed in %.2f seconds.", time.time() - start)
        payload = [
            (run_id, track_id, float(point[0]), float(point[1]))
            for (run_id, track_id, _), point in zip(embeddings, coords, strict=True)
        ]
        upsert_umap(conn, scope=args.scope_label, model=args.model, rows=payload)
    logging.info("UMAP projection stored for scope '%s'.", args.scope_label)


if __name__ == "__main__":
    main()
