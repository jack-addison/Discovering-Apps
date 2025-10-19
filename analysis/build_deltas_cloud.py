#!/usr/bin/env python3
"""Compute snapshot deltas against SQLiteCloud database."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

import pandas as pd
import sqlitecloud

from analysis.build_deltas import (  # type: ignore
    QUERY,
    _parse_best_rank,
    compute_deltas,
    upsert_deltas,
)
from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize app_snapshot_deltas in SQLiteCloud.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--recreate-table", action="store_true")
    return parser.parse_args()


def load_dataframe(uri: str) -> pd.DataFrame:
    with sqlitecloud.connect(uri) as conn:
        df = pd.read_sql_query(QUERY, conn)
    if df.empty:
        raise ValueError("No snapshot data returned from database.")
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    df.sort_values(["track_id", "run_id"], inplace=True)
    df["best_rank"] = df["chart_memberships"].apply(_parse_best_rank).astype("float64")
    return df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    df = load_dataframe(args.connection_uri)
    deltas = compute_deltas(df)
    with sqlitecloud.connect(args.connection_uri) as conn:
        upsert_deltas(conn, deltas, recreate_table=args.recreate_table)


if __name__ == "__main__":
    main()

