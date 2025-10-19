#!/usr/bin/env python3
"""Cloud variant of neighbor computation."""

from __future__ import annotations

import argparse
import logging
import sqlitecloud
from typing import Dict, List, Tuple

import numpy as np

from analysis.build_neighbors import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SIMILARITY,
    ensure_table,
    fetch_embeddings,
    group_embeddings,
    compute_neighbors_for_group,
    upsert_neighbors,
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
        ensure_table(conn)
        embeddings = fetch_embeddings(
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
        upsert_neighbors(conn, model=args.model, groups=results.items())


if __name__ == "__main__":
    main()

