#!/usr/bin/env python3
"""Cluster dissatisfied apps using their embeddings inside Neon."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import psycopg

from ..config import load_settings

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_CLUSTERS = 20
DEFAULT_MAX_ITER = 40
DEFAULT_MIN_CLUSTER = 5
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
    "allows",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster dissatisfied apps in Neon")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id (default: text-embedding-3-small)")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS, help="Number of k-means centroids (default: 20)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="K-means iterations (default: 40)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER, help="Drop clusters smaller than this")
    parser.add_argument("--run-id", type=int, action="append", help="Restrict to specific runs")
    parser.add_argument("--max-apps", type=int, help="Optional cap on embeddings to cluster")
    parser.add_argument("--postgres-dsn", help="Override PROTOTYPE_DATABASE_URL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--top-keywords", type=int, default=6, help="Keywords per cluster (default: 6)")
    parser.add_argument("--scope-label", default="dissatisfied", help="Value for the app_snapshot_clusters.scope column")
    return parser.parse_args()


def fetch_embeddings(
    conn: psycopg.Connection,
    *,
    model: str,
    run_ids: Sequence[int] | None,
    limit: int | None,
) -> List[Tuple[int, int, str, np.ndarray]]:
    filters = ["e.model = %s"]
    params: List[object] = [model]
    if run_ids:
        placeholders = ",".join(["%s"] * len(run_ids))
        filters.append(f"e.run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = " AND ".join(filters)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT e.run_id, e.track_id, s.description, e.embedding_json
        FROM app_snapshot_embeddings e
        JOIN app_snapshot_dissatisfied d
          ON d.run_id = e.run_id AND d.track_id = e.track_id
        JOIN app_snapshots s
          ON s.run_id = e.run_id AND s.track_id = e.track_id
        WHERE {where_clause}
        ORDER BY e.run_id DESC, e.track_id
        {limit_clause}
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    latest: Dict[int, Tuple[int, int, str, np.ndarray]] = {}
    for run_id, track_id, description, embedding_json in rows:
        vector = np.array(json.loads(embedding_json), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        normalized = vector / norm
        existing = latest.get(track_id)
        if existing is None or run_id > existing[0]:
            latest[track_id] = (run_id, track_id, description or "", normalized)
    return list(latest.values())


def kmeans(matrix: np.ndarray, clusters: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_samples = matrix.shape[0]
    clusters = min(clusters, max(1, n_samples))
    if clusters == 1:
        return np.zeros(n_samples, dtype=int), matrix[:1]
    indices = rng.choice(n_samples, clusters, replace=False)
    centroids = matrix[indices]
    for _ in range(max_iter):
        distances = 1 - (matrix @ centroids.T)
        assignments = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for idx in range(clusters):
            members = matrix[assignments == idx]
            if len(members) == 0:
                new_centroids[idx] = centroids[idx]
            else:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm == 0:
                    new_centroids[idx] = centroids[idx]
                else:
                    new_centroids[idx] = centroid / norm
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return assignments, centroids


def extract_keywords(descriptions: Iterable[str], top_k: int, stopwords: Sequence[str]) -> List[str]:
    pattern = re.compile(r"[A-Za-z]{3,}")
    counter: Counter[str] = Counter()
    stop = set(stopwords)
    for text in descriptions:
        tokens = [tok.lower() for tok in pattern.findall(text)]
        filtered = [tok for tok in tokens if tok not in stop]
        counter.update(filtered)
    return [word for word, _ in counter.most_common(top_k)]


def upsert_clusters(
    conn: psycopg.Connection,
    *,
    scope: str,
    model: str,
    assignments: np.ndarray,
    centroids: np.ndarray,
    rows: Sequence[Tuple[int, int, str, np.ndarray]],
    min_size: int,
    top_keywords: int,
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM app_snapshot_cluster_members WHERE cluster_id IN (SELECT id FROM app_snapshot_clusters WHERE scope = %s AND model = %s)",
            (scope, model),
        )
        cur.execute(
            "DELETE FROM app_snapshot_clusters WHERE scope = %s AND model = %s",
            (scope, model),
        )
        grouped: Dict[int, List[int]] = defaultdict(list)
        for idx, cluster_idx in enumerate(assignments):
            grouped[int(cluster_idx)].append(idx)

        for cluster_idx, member_indices in grouped.items():
            if len(member_indices) < min_size:
                continue
            member_rows = [rows[i] for i in member_indices]
            keywords = extract_keywords((row[2] for row in member_rows), top_k=top_keywords, stopwords=DEFAULT_STOPWORDS)
            label = ", ".join(keywords) if keywords else "Mixed apps"
            metrics_params = []
            placeholders = []
            for run_id, track_id, *_ in member_rows:
                placeholders.append("(%s, %s)")
                metrics_params.extend([run_id, track_id])
            metrics_sql = f"""
                SELECT
                    AVG(s.average_user_rating) AS avg_rating,
                    AVG(s.user_rating_count) AS avg_volume
                FROM app_snapshots s
                WHERE (s.run_id, s.track_id) IN ({','.join(placeholders)})
            """
            cur.execute(metrics_sql, metrics_params)
            metric_row = cur.fetchone()
            cluster_avg_rating = metric_row[0]
            cluster_avg_volume = metric_row[1]
            cur.execute(
                """
                INSERT INTO app_snapshot_clusters (
                    scope, model, label, keywords_json, size, avg_success, avg_build, avg_demand, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    scope,
                    model,
                    label,
                    json.dumps(keywords),
                    len(member_rows),
                    cluster_avg_rating,
                    cluster_avg_volume,
                    None,
                    timestamp,
                ),
            )
            cluster_id = cur.fetchone()[0]
            centroid = centroids[cluster_idx]
            member_payload = []
            for run_id, track_id, _, vector in member_rows:
                distance = float(1 - np.dot(vector, centroid))
                member_payload.append((cluster_id, run_id, track_id, distance))
            cur.executemany(
                """
                INSERT INTO app_snapshot_cluster_members (cluster_id, run_id, track_id, distance)
                VALUES (%s, %s, %s, %s)
                """,
                member_payload,
            )
    conn.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn

    with psycopg.connect(dsn) as conn:
        run_ids = sorted(set(args.run_id)) if args.run_id else None
        embeddings = fetch_embeddings(
            conn,
            model=args.model,
            run_ids=run_ids,
            limit=args.max_apps,
        )
        logging.info("Loaded %s embeddings", len(embeddings))
        if len(embeddings) < max(args.min_cluster_size, 2):
            logging.warning("Not enough embeddings to cluster.")
            return
        matrix = np.vstack([row[3] for row in embeddings])
        assignments, centroids = kmeans(matrix, clusters=args.clusters, max_iter=args.max_iter)
        upsert_clusters(
            conn,
            scope=args.scope_label,
            model=args.model,
            assignments=assignments,
            centroids=centroids,
            rows=embeddings,
            min_size=args.min_cluster_size,
            top_keywords=args.top_keywords,
        )
    logging.info("Clustering complete for scope '%s'", args.scope_label)


if __name__ == "__main__":
    main()
