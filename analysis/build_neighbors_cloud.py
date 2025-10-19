#!/usr/bin/env python3
"""Cloud variant of neighbor computation."""

from __future__ import annotations

import argparse
import json
import logging
import sqlitecloud
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_neighbors import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SIMILARITY,
    EmbeddingRecord,
    group_embeddings,
    compute_neighbors_for_group,
)
from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute embedding neighbours in SQLiteCloud.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--run-id", type=int, action="append")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--max-apps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with sqlitecloud.connect(args.connection_uri) as conn:
        ensure_table_cloud(conn)
        embeddings = fetch_embeddings_cloud(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        if not embeddings:
            logging.info("No embeddings available.")
            return
        grouped = group_embeddings(embeddings, all_runs=args.all_runs)
        results = {}
        for group_key, rows in grouped.items():
            neighbours = compute_neighbors_for_group(
                rows,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
            )
            results[group_key] = neighbours
        upsert_neighbors_cloud(conn, model=args.model, groups=results.items())


def ensure_table_cloud(connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_snapshot_neighbors (
            run_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            neighbor_run_id INTEGER NOT NULL,
            neighbor_track_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            similarity REAL NOT NULL,
            rank INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, track_id, neighbor_run_id, neighbor_track_id, model)
        )
        """
    )
    connection.commit()
    cursor.close()


def fetch_embeddings_cloud(
    connection,
    *,
    model: str,
    run_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[EmbeddingRecord]:
    filters = ["model = ?"]
    params: List[Any] = [model]
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = "WHERE " + " AND ".join(filters)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT run_id, track_id, model, embedding_json
        FROM app_snapshot_embeddings
        {where_clause}
        ORDER BY run_id DESC, track_id
        {limit_clause}
    """
    cursor = connection.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    embeddings: List[EmbeddingRecord] = []
    for row in rows:
        record = dict(zip(columns, row))
        vector = np.array(json.loads(record["embedding_json"]), dtype=np.float32)
        if np.linalg.norm(vector) == 0:
            continue
        embeddings.append(
            EmbeddingRecord(
                run_id=record["run_id"],
                track_id=record["track_id"],
                model=record["model"],
                vector=vector,
            )
        )
    return embeddings


def upsert_neighbors_cloud(
    connection,
    *,
    model: str,
    groups: Iterable[Tuple[int, List[Tuple[EmbeddingRecord, List[Tuple[EmbeddingRecord, float]]]]]],
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor = connection.cursor()
    for group_key, entries in groups:
        if group_key != 0:
            cursor.execute(
                "DELETE FROM app_snapshot_neighbors WHERE run_id = ? AND model = ?",
                (group_key, model),
            )
        else:
            cursor.execute(
                "DELETE FROM app_snapshot_neighbors WHERE model = ?",
                (model,),
            )
        rows_to_insert = []
        for base_record, neighbours in entries:
            for rank, (neighbour_record, similarity) in enumerate(neighbours, start=1):
                rows_to_insert.append(
                    (
                        base_record.run_id,
                        base_record.track_id,
                        neighbour_record.run_id,
                        neighbour_record.track_id,
                        model,
                        similarity,
                        rank,
                        timestamp,
                    )
                )
        if rows_to_insert:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO app_snapshot_neighbors (
                    run_id,
                    track_id,
                    neighbor_run_id,
                    neighbor_track_id,
                    model,
                    similarity,
                    rank,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            connection.commit()
    cursor.close()


if __name__ == "__main__":
    main()
