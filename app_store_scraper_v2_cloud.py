#!/usr/bin/env python3
"""SQLiteCloud variant of the Stage 1 snapshot scraper."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import requests
import sqlitecloud

from app_store_scraper_v2 import (  # type: ignore
    DEFAULT_HEADERS,
    ITUNES_CHART_PATHS,
    RSS_COLLECTION_URL,
    SEARCH_API_URL,
    LOOKUP_API_URL,
    GENRE_TREE_URL,
    bounded_limit,
    fetch_category_chart_ids,
    fetch_category_map,
    fetch_collection_ids,
    fetch_search_results,
    insert_rankings,
    insert_run,
    insert_snapshots,
    lookup_apps,
    normalize_snapshot,
    build_category_memberships,
)

from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Apple App Store metadata and write snapshots to SQLiteCloud."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--search-term", help="Keyword to search for within the App Store.")
    source_group.add_argument(
        "--collection",
        choices=["top-free", "top-paid", "top-grossing"],
        help="Fetch the specified top chart collection.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum apps to fetch (1-200). Default: 100.")
    parser.add_argument("--country", default="us", help="Two-letter country code for the App Store.")
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="When set with --collection, iterate every category to capture rankings.",
    )
    parser.add_argument(
        "--connection-uri",
        default=CONNECTION_URI,
        help="SQLiteCloud connection URI (defaults to cloud_config.CONNECTION_URI).",
    )
    parser.add_argument("--lookup-batch-size", type=int, default=100, help="Batch size when requesting metadata.")
    parser.add_argument("--note", help="Optional note stored with this scrape run for later reference.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    args = parser.parse_args()
    if args.all_categories and not args.collection:
        parser.error("--all-categories requires --collection.")
    return args


def collect_records(args: argparse.Namespace):
    if args.search_term:
        raw_records = fetch_search_results(args.search_term, args.country, args.limit)
        membership_map: Dict[int, List[Dict[str, object]]] = {}
        ranking_rows: List[Dict[str, object]] = []
        source = "search"
    elif args.all_categories:
        membership_map, ranking_rows = build_category_memberships(
            args.collection,
            args.country,
            args.limit,
        )
        raw_records = lookup_apps(list(membership_map.keys()), args.country, args.lookup_batch_size)
        source = "collection-all-categories"
    else:
        ids = fetch_collection_ids(args.collection, args.country, args.limit)
        membership_map = {
            int(app_id): [
                {
                    "chart_type": args.collection,
                    "category_id": None,
                    "category_name": args.collection,
                    "rank": rank,
                }
            ]
            for rank, app_id in enumerate(ids, start=1)
        }
        ranking_rows = [
            {
                "track_id": int(app_id),
                "chart_type": args.collection,
                "category_id": None,
                "category_name": args.collection,
                "rank": rank,
            }
            for rank, app_id in enumerate(ids, start=1)
        ]
        raw_records = lookup_apps(ids, args.country, args.lookup_batch_size)
        source = "collection"
    return raw_records, membership_map, ranking_rows, source


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    raw_records, membership_map, ranking_rows, source = collect_records(args)
    if not raw_records:
        logging.warning("No app metadata retrieved.")
        return

    scraped_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    connection = sqlitecloud.connect(args.connection_uri)

    run_id = insert_run(
        connection,
        source=source,
        country=args.country,
        collection=args.collection,
        search_term=args.search_term,
        limit_requested=args.limit,
        all_categories=args.all_categories,
        note=args.note,
    )
    logging.info("Created scrape run %s", run_id)

    snapshots: List[Dict[str, object]] = []
    for record in raw_records:
        track_id = record.get("trackId")
        if not track_id:
            continue
        memberships = membership_map.get(int(track_id))
        snapshot = normalize_snapshot(record, run_id, memberships, scraped_at)
        snapshots.append(snapshot)

    insert_snapshots(connection, snapshots)
    insert_rankings(connection, run_id, ranking_rows)
    connection.close()
    logging.info("Persisted %d app snapshots (run %s) to SQLiteCloud", len(snapshots), run_id)


if __name__ == "__main__":
    main()

