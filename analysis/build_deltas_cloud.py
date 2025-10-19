#!/usr/bin/env python3
"""Compute snapshot deltas against SQLiteCloud database."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlitecloud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_deltas import (  # type: ignore
    QUERY,
    _parse_best_rank,
    compute_deltas,
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
        write_output_cloud(deltas, conn, recreate=args.recreate_table)


def write_output_cloud(df: pd.DataFrame, connection, *, recreate: bool) -> None:
    ensure_table_cloud(connection, recreate)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM app_snapshot_deltas")
    columns = [
        "run_id",
        "run_created_at",
        "track_id",
        "name",
        "category",
        "price",
        "currency",
        "average_user_rating",
        "user_rating_count",
        "build_time_estimate",
        "success_score",
        "success_reasoning",
        "best_rank",
        "description",
        "version",
        "developer",
        "prev_run_id",
        "prev_run_created_at",
        "prev_success_score",
        "prev_build_time",
        "prev_rating",
        "prev_rating_count",
        "prev_price",
        "prev_currency",
        "prev_rank",
        "is_new_track",
        "delta_success",
        "delta_build_time",
        "delta_rating",
        "delta_rating_count",
        "delta_price",
        "delta_rank",
        "price_changed",
        "days_since_prev",
    ]
    insert_sql = (
        "INSERT INTO app_snapshot_deltas ("
        + ",".join(columns)
        + ") VALUES ("
        + ",".join([":" + col for col in columns])
        + ")"
    )
    working = df[columns].copy()
    for col in ["run_created_at", "prev_run_created_at"]:
        if col in working.columns:
            working[col] = working[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )
    # ensure pandas NA converted to None for other fields
    working = working.replace({pd.NA: None})
    records = working.to_dict("records")
    chunk = 500
    for start in range(0, len(records), chunk):
        batch = records[start : start + chunk]
        cursor.executemany(insert_sql, batch)
        connection.commit()
    cursor.close()


def ensure_table_cloud(connection, recreate: bool) -> None:
    cursor = connection.cursor()
    if recreate:
        cursor.execute("DROP TABLE IF EXISTS app_snapshot_deltas")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_snapshot_deltas (
            run_id INTEGER NOT NULL,
            run_created_at TEXT,
            track_id INTEGER NOT NULL,
            name TEXT,
            category TEXT,
            price REAL,
            currency TEXT,
            average_user_rating REAL,
            user_rating_count INTEGER,
            build_time_estimate REAL,
            success_score REAL,
            success_reasoning TEXT,
            best_rank REAL,
            description TEXT,
            version TEXT,
            developer TEXT,
            prev_run_id INTEGER,
            prev_run_created_at TEXT,
            prev_success_score REAL,
            prev_build_time REAL,
            prev_rating REAL,
            prev_rating_count REAL,
            prev_price REAL,
            prev_currency TEXT,
            prev_rank REAL,
            is_new_track INTEGER,
            delta_success REAL,
            delta_build_time REAL,
            delta_rating REAL,
            delta_rating_count REAL,
            delta_price REAL,
            delta_rank REAL,
            price_changed INTEGER,
            days_since_prev REAL,
            PRIMARY KEY (run_id, track_id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_track ON app_snapshot_deltas(track_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_run ON app_snapshot_deltas(run_id)")
    connection.commit()
    cursor.close()


if __name__ == "__main__":
    main()
