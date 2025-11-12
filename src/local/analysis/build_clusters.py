#!/usr/bin/env python3
"""Cluster app snapshots using stored embeddings and keyword labels."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_CLUSTERS = 20
DEFAULT_MAX_ITER = 40
DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "your",
    "that",
    "from",
    "this",
    "you",
    "have",
    "are",
    "app",
    "apps",
    "into",
    "will",
    "they",
    "their",
    "about",
    "more",
    "can",
    "when",
    "while",
    "where",
    "what",
    "make",
    "made",
    "easy",
    "using",
    "best",
    "easy",
    "easy",
    "your",
    "our",
    "help",
    "helping",
    "new",
    "free",
    "get",
    "keep",
    "also",
    "just",
    "any",
    "every",
    "each",
    "one",
    "now",
    "use",
    "using",
    "allows",
    "allow",
    "includes",
    "including",
    "features",
    "feature",
    "build",
    "designed",
    "designed",
    "designed",
    "designed",
    "experience",
}


@dataclass
class EmbeddingRow:
    run_id: int
    track_id: int
    description: str
    vector: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster snapshot embeddings and store labelled groups."
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
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Number of clusters to create (default: {DEFAULT_CLUSTERS}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum k-means iterations (default: {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        action="append",
        help="Restrict clustering to specific run IDs (repeatable).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Cluster across all runs instead of separately per run.",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        help="Optional limit on number of embeddings to cluster (for smoke tests).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Discard clusters smaller than this size (default: 5).",
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=6,
        help="Number of keywords to keep for each cluster label (default: 6).",
    )
    args = parser.parse_args()

    if args.clusters <= 0:
        parser.error("--clusters must be positive.")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive.")
    if args.min_cluster_size <= 1:
        parser.error("--min-cluster-size must be greater than 1.")

    return args


def ensure_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_snapshot_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            model TEXT NOT NULL,
            label TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            size INTEGER NOT NULL,
            avg_success REAL,
            avg_build REAL,
            avg_demand REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_snapshot_cluster_members (
            cluster_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            distance REAL NOT NULL,
            PRIMARY KEY (cluster_id, run_id, track_id),
            FOREIGN KEY (cluster_id) REFERENCES app_snapshot_clusters(id)
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
) -> List[EmbeddingRow]:
    filters = ["model = ?"]
    params: List[object] = [model]
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = f"WHERE {' AND '.join(filters)}"
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
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, params).fetchall()

    latest_by_track: Dict[int, EmbeddingRow] = {}
    for row in rows:
        vector = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        record = EmbeddingRow(
            run_id=row["run_id"],
            track_id=row["track_id"],
            description=row["description"] or "",
            vector=vector / norm,
        )
        existing = latest_by_track.get(record.track_id)
        if existing is None or record.run_id > existing.run_id:
            latest_by_track[record.track_id] = record

    return list(latest_by_track.values())


def kmeans(
    matrix: np.ndarray,
    *,
    clusters: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_samples = matrix.shape[0]
    if clusters >= n_samples:
        clusters = max(1, n_samples - 1)
    indices = rng.choice(n_samples, clusters, replace=False)
    centroids = matrix[indices]

    for _ in range(max_iter):
        distances = 1 - (matrix @ centroids.T)
        assignments = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(clusters, dtype=int)
        for idx in range(clusters):
            members = matrix[assignments == idx]
            if len(members) == 0:
                new_centroids[idx] = centroids[idx]
            else:
                new_centroids[idx] = members.mean(axis=0)
                counts[idx] = len(members)
        new_centroids = new_centroids / np.linalg.norm(new_centroids, axis=1, keepdims=True)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return assignments, centroids


def extract_keywords(
    descriptions: Iterable[str],
    *,
    top_k: int,
    stopwords: Sequence[str],
) -> List[str]:
    pattern = re.compile(r"[A-Za-z]{3,}")
    counter: Counter[str] = Counter()
    stop = set(stopwords)
    for text in descriptions:
        tokens = [tok.lower() for tok in pattern.findall(text)]
        filtered = [tok for tok in tokens if tok not in stop]
        counter.update(filtered)
    return [word for word, _ in counter.most_common(top_k)]


def compute_scope(run_ids: Optional[Sequence[int]], all_runs: bool) -> str:
    if all_runs or not run_ids:
        return "all"
    return "runs:" + ",".join(str(rid) for rid in sorted(set(run_ids)))


def upsert_clusters(
    connection: sqlite3.Connection,
    *,
    scope: str,
    model: str,
    assignments: np.ndarray,
    centroids: np.ndarray,
    rows: Sequence[EmbeddingRow],
    min_size: int,
    top_keywords: int,
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    connection.execute(
        "DELETE FROM app_snapshot_cluster_members WHERE cluster_id IN (SELECT id FROM app_snapshot_clusters WHERE scope = ? AND model = ?)",
        (scope, model),
    )
    connection.execute(
        "DELETE FROM app_snapshot_clusters WHERE scope = ? AND model = ?",
        (scope, model),
    )

    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, cluster_idx in enumerate(assignments):
        grouped[int(cluster_idx)].append(idx)

    cursor = connection.cursor()
    for cluster_idx, member_indices in grouped.items():
        if len(member_indices) < min_size:
            continue
        member_rows = [rows[i] for i in member_indices]
        keywords = extract_keywords(
            (row.description for row in member_rows),
            top_k=top_keywords,
            stopwords=DEFAULT_STOPWORDS,
        )
        label = ", ".join(keywords) if keywords else "Mixed apps"
        metrics = connection.execute(
            """
            SELECT
                AVG(success_score) AS avg_success,
                AVG(build_time_estimate) AS avg_build,
                AVG(user_rating_count * (5 - average_user_rating)) AS avg_demand
            FROM app_snapshots
            WHERE (run_id, track_id) IN (%s)
            """
            % ",".join(["(?, ?)"] * len(member_rows)),
            [value for row in member_rows for value in (row.run_id, row.track_id)],
        ).fetchone()
        cursor.execute(
            """
            INSERT INTO app_snapshot_clusters (
                scope, model, label, keywords_json, size, avg_success, avg_build, avg_demand, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope,
                model,
                label or "Unnamed cluster",
                json.dumps(keywords),
                len(member_rows),
                metrics[0],
                metrics[1],
                metrics[2],
                timestamp,
            ),
        )
        cluster_id = cursor.lastrowid
        centroid = centroids[cluster_idx]
        for row in member_rows:
            distance = float(1 - np.dot(row.vector, centroid))
            cursor.execute(
                """
                INSERT INTO app_snapshot_cluster_members (cluster_id, run_id, track_id, distance)
                VALUES (?, ?, ?, ?)
                """,
                (cluster_id, row.run_id, row.track_id, distance),
            )

    connection.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with sqlite3.connect(args.db_path) as conn:
        ensure_tables(conn)
        embeddings = fetch_embeddings(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        logging.info("Loaded %d embeddings", len(embeddings))
        if len(embeddings) < max(args.min_cluster_size, 2):
            logging.warning("Not enough embeddings to cluster. Aborting.")
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

    logging.info("Cluster build complete for scope '%s'", scope)


if __name__ == "__main__":
    main()
