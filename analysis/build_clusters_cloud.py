#!/usr/bin/env python3
"""Cloud-aware clustering pipeline."""

from __future__ import annotations

import argparse
import logging
import sqlitecloud
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_clusters import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_CLUSTERS,
    DEFAULT_MAX_ITER,
    DEFAULT_STOPWORDS,
    parse_args as base_parse,  # not used
    ensure_tables,
    fetch_embeddings,
    kmeans,
    compute_scope,
    upsert_clusters,
)
from cloud_config import CONNECTION_URI


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

    with sqlitecloud.connect(args.connection_uri) as conn:
        ensure_tables(conn)
        embeddings = fetch_embeddings(
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


if __name__ == "__main__":
    import numpy as np  # local import to avoid dependency for help text

    main()
