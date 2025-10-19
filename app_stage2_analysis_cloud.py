#!/usr/bin/env python3
"""Cloud variant of Stage 2 scoring script."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

import sqlitecloud
from openai import OpenAI

from app_stage2_analysis import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_SLEEP_SECONDS,
    AppRecord,
    ensure_columns,
    fetch_apps,
    process_apps,
)
from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate build_time_estimate and success_score columns in SQLiteCloud using the OpenAI API."
    )
    parser.add_argument(
        "--connection-uri",
        default=CONNECTION_URI,
        help="SQLiteCloud connection URI (defaults to cloud_config.CONNECTION_URI).",
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

    with sqlitecloud.connect(args.connection_uri) as conn:
        ensure_columns(conn)
        apps = fetch_apps(
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


if __name__ == "__main__":
    main()
