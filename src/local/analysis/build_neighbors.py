#!/usr/bin/env python3
"""Compute nearest-neighbour relationships between app snapshots using stored embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.75


@dataclass(frozen=True)
class EmbeddingRecord:
    run_id: int
    track_id: int
    model: str
    vector: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the app_snapshot_neighbors table with cosine-similarity matches."
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
        help=f"Embedding model identifier to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of neighbours to keep per app (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"Minimum cosine similarity required to keep a neighbour (default: {DEFAULT_MIN_SIMILARITY}).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        action="append",
        help="Restrict processing to one or more run IDs (repeatable).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Pool embeddings across all run IDs instead of computing neighbours within each run.",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        help="Optional limit on the number of rows to process (useful for smoke tests).",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if not (0 < args.min_similarity <= 1):
        parser.error("--min-similarity must be in (0, 1].")

    return args


def ensure_table(connection: sqlite3.Connection) -> None:
    connection.execute(
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


def fetch_embeddings(
    connection: sqlite3.Connection,
    *,
    model: str,
    run_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[EmbeddingRecord]:
    filters = ["model = ?"]
    params: List[object] = [model]
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = f"WHERE {' AND '.join(filters)}"
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT run_id, track_id, model, embedding_json
        FROM app_snapshot_embeddings
        {where_clause}
        ORDER BY run_id DESC, track_id
        {limit_clause}
    """
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, params).fetchall()
    embeddings: List[EmbeddingRecord] = []
    for row in rows:
        vector = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
        if np.linalg.norm(vector) == 0:
            continue
        embeddings.append(
            EmbeddingRecord(
                run_id=row["run_id"],
                track_id=row["track_id"],
                model=row["model"],
                vector=vector,
            )
        )
    return embeddings


def group_embeddings(
    embeddings: Sequence[EmbeddingRecord], *, all_runs: bool
) -> Dict[int, List[EmbeddingRecord]]:
    if all_runs:
        return {0: list(embeddings)}

    grouped: Dict[int, List[EmbeddingRecord]] = {}
    for record in embeddings:
        grouped.setdefault(record.run_id, []).append(record)
    return grouped


def compute_neighbors_for_group(
    records: Sequence[EmbeddingRecord],
    *,
    top_k: int,
    min_similarity: float,
) -> List[Tuple[EmbeddingRecord, List[Tuple[EmbeddingRecord, float]]]]:
    if len(records) <= 1:
        return []

    matrix = np.vstack([r.vector for r in records])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normalised = matrix / norms
    similarities = matrix_normalised @ matrix_normalised.T

    neighbours: List[Tuple[EmbeddingRecord, List[Tuple[EmbeddingRecord, float]]]] = []
    for idx, record in enumerate(records):
        sims = similarities[idx]
        sims[idx] = -1  # exclude self
        candidate_indices = np.argpartition(sims, -top_k)[-top_k:]
        sorted_indices = candidate_indices[np.argsort(-sims[candidate_indices])]
        selected: List[Tuple[EmbeddingRecord, float]] = []
        for neighbour_idx in sorted_indices:
            similarity = float(sims[neighbour_idx])
            if similarity < min_similarity:
                continue
            selected.append((records[neighbour_idx], similarity))
        if selected:
            neighbours.append((record, selected))
    return neighbours


def upsert_neighbors(
    connection: sqlite3.Connection,
    *,
    model: str,
    groups: Iterable[Tuple[int, List[Tuple[EmbeddingRecord, List[Tuple[EmbeddingRecord, float]]]]]],
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for group_key, entries in groups:
        if group_key != 0:
            connection.execute(
                "DELETE FROM app_snapshot_neighbors WHERE run_id = ? AND model = ?",
                (group_key, model),
            )
        else:
            connection.execute(
                "DELETE FROM app_snapshot_neighbors WHERE model = ?",
                (model,),
            )
        for base_record, neighbours in entries:
            for rank, (neighbour_record, similarity) in enumerate(neighbours, start=1):
                connection.execute(
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
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        base_record.run_id,
                        base_record.track_id,
                        neighbour_record.run_id,
                        neighbour_record.track_id,
                        model,
                        similarity,
                        rank,
                        timestamp,
                    ),
                )
    connection.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with sqlite3.connect(args.db_path) as conn:
        ensure_table(conn)
        embeddings = fetch_embeddings(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        logging.info("Loaded %d embeddings for model %s", len(embeddings), args.model)
        if not embeddings:
            logging.info("Nothing to process.")
            return

        grouped = group_embeddings(embeddings, all_runs=args.all_runs)
        logging.info("Computing neighbours for %d group(s)", len(grouped))

        results: Dict[int, List[Tuple[EmbeddingRecord, List[Tuple[EmbeddingRecord, float]]]]] = {}
        for group_key, records in grouped.items():
            start = time.time()
            neighbours = compute_neighbors_for_group(
                records,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
            )
            elapsed = time.time() - start
            logging.info(
                "Group %s: %d embeddings → %d neighbour sets (%.2fs)",
                "all" if group_key == 0 else group_key,
                len(records),
                len(neighbours),
                elapsed,
            )
            results[group_key] = neighbours

        upsert_neighbors(
            conn,
            model=args.model,
            groups=results.items(),
        )
        logging.info("Neighbour table updated.")


if __name__ == "__main__":
    main()
