#!/usr/bin/env python3
"""Cloud-aware clustering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.local.analysis.build_clusters import (
    DEFAULT_MODEL,
    DEFAULT_CLUSTERS,
    DEFAULT_MAX_ITER,
    DEFAULT_STOPWORDS,
    ensure_tables,
    EmbeddingRow,
    kmeans,
    compute_scope,
    upsert_clusters,
)
from config.cloud import CONNECTION_URI, connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster embeddings inside SQLiteCloud.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--run-id", type=int, action="append")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--max-apps", type=int)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--top-keywords", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with connect(uri=args.connection_uri) as conn:
        ensure_tables(conn)
        embeddings = fetch_embeddings_cloud(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        if len(embeddings) < max(args.min_cluster_size, 2):
            logging.warning("Not enough embeddings to cluster.")
            return

        scope = compute_scope(args.run_id, args.all_runs)
        matrix = np.vstack([row.vector for row in embeddings])
        clusters = min(args.clusters, len(embeddings))
        assignments, centroids = kmeans(matrix, clusters=clusters, max_iter=args.max_iter)
        upsert_clusters(
            conn,
            scope=scope,
            model=args.model,
            assignments=assignments,
            centroids=centroids,
            rows=embeddings,
            min_size=args.min_cluster_size,
            top_keywords=args.top_keywords,
        )


def fetch_embeddings_cloud(
    connection,
    *,
    model: str,
    run_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[EmbeddingRow]:
    filters = ["e.model = ?"]
    params: List[Any] = [model]
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"e.run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = "WHERE " + " AND ".join(filters)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT e.run_id, e.track_id, s.description, e.embedding_json
        FROM app_snapshot_embeddings AS e
        JOIN app_snapshots AS s
          ON s.run_id = e.run_id
         AND s.track_id = e.track_id
        {where_clause}
        ORDER BY e.run_id DESC, e.track_id
        {limit_clause}
    """
    cursor = connection.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    latest_by_track: Dict[int, EmbeddingRow] = {}
    for row in rows:
        record = dict(zip(columns, row))
        vector = np.array(json.loads(record["embedding_json"]), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        entry = EmbeddingRow(
            run_id=record["run_id"],
            track_id=record["track_id"],
            description=record.get("description") or "",
            vector=vector / norm,
        )
        existing = latest_by_track.get(entry.track_id)
        if existing is None or entry.run_id > existing.run_id:
            latest_by_track[entry.track_id] = entry
    return list(latest_by_track.values())


if __name__ == "__main__":
    import numpy as np  # local import to avoid dependency for help text

    main()
