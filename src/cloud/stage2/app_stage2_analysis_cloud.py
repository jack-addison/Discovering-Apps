#!/usr/bin/env python3
"""Cloud variant of Stage 2 scoring script."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, List, Optional, Sequence

from openai import OpenAI

from src.local.stage2.app_stage2_analysis import (
    DEFAULT_MODEL,
    DEFAULT_SLEEP_SECONDS,
    AppRecord,
    ensure_columns,
    process_apps,
)
from config.cloud import CONNECTION_URI, connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate build_time_estimate and success_score columns in SQLiteCloud using the OpenAI API."
    )
    parser.add_argument(
        "--connection-uri",
        default=CONNECTION_URI,
        help="SQLiteCloud connection URI (defaults to config.cloud.CONNECTION_URI).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=int, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--batch-progress", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-apps", type=int)
    parser.add_argument("--run-id", type=int, action="append")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Environment variable OPENAI_API_KEY must be set before running this script.")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with connect(uri=args.connection_uri) as conn:
        ensure_columns(conn)
        apps = fetch_apps_cloud(
            conn,
            include_existing=args.force,
            limit=args.max_apps,
            run_ids=args.run_id,
        )
        if not apps:
            logging.info("No apps requiring Stage 2 scoring.")
            return
        logging.info("Scoring %d apps against %s", len(apps), args.model)
        process_apps(
            conn,
            apps,
            model=args.model,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            batch_progress=args.batch_progress,
        )


def fetch_apps_cloud(
    connection,
    *,
    include_existing: bool,
    limit: Optional[int],
    run_ids: Optional[Sequence[int]],
) -> List[AppRecord]:
    filters: List[str] = []
    params: List[Any] = []
    if not include_existing:
        filters.append(
            "(build_time_estimate IS NULL OR success_score IS NULL OR success_reasoning IS NULL)"
        )
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        filters.append(f"run_id IN ({placeholders})")
        params.extend(run_ids)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT
            run_id,
            track_id,
            name,
            primary_genre_name AS category,
            average_user_rating AS review_score,
            user_rating_count AS number_of_ratings,
            user_rating_count AS number_of_downloads,
            price,
            currency,
            seller_name AS developer,
            description,
            language_codes,
            chart_memberships,
            build_time_estimate,
            success_score,
            success_reasoning
        FROM app_snapshots
        {where_clause}
        ORDER BY run_id DESC, track_id
        {limit_clause}
    """
    cursor = connection.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    apps: List[AppRecord] = []
    for row in rows:
        data = dict(zip(columns, row))
        apps.append(
            AppRecord(
                run_id=data["run_id"],
                track_id=data["track_id"],
                name=data["name"],
                category=data.get("category"),
                review_score=data.get("review_score"),
                number_of_ratings=data.get("number_of_ratings"),
                number_of_downloads=data.get("number_of_downloads"),
                price=data.get("price"),
                currency=data.get("currency"),
                developer=data.get("developer"),
                description=data.get("description") or "",
                language_codes=data.get("language_codes"),
                chart_memberships=data.get("chart_memberships"),
                existing_build_time=data.get("build_time_estimate"),
                existing_success_score=data.get("success_score"),
                existing_success_reasoning=data.get("success_reasoning"),
            )
        )
    return apps


if __name__ == "__main__":
    main()
