#!/usr/bin/env python3
"""Generate per-snapshot deltas and features for predictive analysis."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"


QUERY = """
SELECT
    s.run_id,
    COALESCE(sr.created_at, s.scraped_at) AS run_created_at,
    s.track_id,
    s.name,
    s.primary_genre_name AS category,
    s.price,
    s.currency,
    s.average_user_rating,
    s.user_rating_count,
    s.build_time_estimate,
    s.success_score,
    s.success_reasoning,
    s.chart_memberships,
    s.description,
    s.version,
    s.seller_name AS developer
FROM app_snapshots s
LEFT JOIN scrape_runs sr ON sr.id = s.run_id
ORDER BY s.track_id, s.run_id
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize app-level deltas between snapshot runs into the database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to app_store_apps_v2.db (default: exports/app_store_apps_v2.db)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--recreate-table",
        action="store_true",
        help="Drop and recreate the app_snapshot_deltas table before inserting rows.",
    )
    return parser.parse_args()


def load_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    logging.info("Loading snapshots from %s", db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(QUERY, conn)
    if df.empty:
        raise ValueError("No snapshot data returned from database.")
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    df.sort_values(["track_id", "run_id"], inplace=True)

    # extract best rank per snapshot for baseline comparison
    df["best_rank"] = (
        df["chart_memberships"]
        .apply(lambda raw: _parse_best_rank(raw))
        .astype("float64")
    )
    return df


def _parse_best_rank(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        memberships = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(memberships, dict):
        memberships = [memberships]
    ranks = [entry.get("rank") for entry in memberships if isinstance(entry, dict) and entry.get("rank") is not None]
    if not ranks:
        return None
    return min(ranks)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("track_id")
    df["prev_run_id"] = grouped["run_id"].shift(1)
    df["prev_run_created_at"] = grouped["run_created_at"].shift(1)
    df["prev_success_score"] = grouped["success_score"].shift(1)
    df["prev_build_time"] = grouped["build_time_estimate"].shift(1)
    df["prev_rating"] = grouped["average_user_rating"].shift(1)
    df["prev_rating_count"] = grouped["user_rating_count"].shift(1)
    df["prev_price"] = grouped["price"].shift(1)
    df["prev_currency"] = grouped["currency"].shift(1)
    df["prev_rank"] = grouped["best_rank"].shift(1)
    df["is_new_track"] = df["prev_run_id"].isna()
    df["delta_success"] = df["success_score"] - df["prev_success_score"]
    df["delta_build_time"] = df["build_time_estimate"] - df["prev_build_time"]
    df["delta_rating"] = df["average_user_rating"] - df["prev_rating"]
    df["delta_rating_count"] = df["user_rating_count"] - df["prev_rating_count"]
    df["delta_price"] = df["price"] - df["prev_price"]
    df["delta_rank"] = df["best_rank"] - df["prev_rank"]
    df["price_changed"] = (df["price"] != df["prev_price"]) | (df["currency"] != df["prev_currency"])

    # days since previous run for same app
    df["days_since_prev"] = (
        df["run_created_at"] - df["prev_run_created_at"]
    ).dt.total_seconds() / (24 * 3600)

    df["is_new_track"] = df["is_new_track"].astype(int)
    df["price_changed"] = df["price_changed"].astype(int)

    return df


def ensure_table(connection: sqlite3.Connection, recreate: bool) -> None:
    if recreate:
        logging.info("Dropping existing app_snapshot_deltas table")
        connection.execute("DROP TABLE IF EXISTS app_snapshot_deltas")

    connection.execute(
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
    connection.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_track ON app_snapshot_deltas(track_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_run ON app_snapshot_deltas(run_id)")
    connection.commit()


def write_output(df: pd.DataFrame, connection: sqlite3.Connection, recreate: bool) -> None:
    ensure_table(connection, recreate)
    logging.info("Clearing and inserting %d rows into app_snapshot_deltas", len(df))
    connection.execute("DELETE FROM app_snapshot_deltas")
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
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in delta DataFrame: {missing_cols}")
    df[columns].to_sql("app_snapshot_deltas", connection, if_exists="append", index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    connection = sqlite3.connect(args.db_path)
    df = load_dataframe(args.db_path)
    feature_df = compute_deltas(df)
    write_output(feature_df, connection, args.recreate_table)
    connection.close()

    logging.info(
        "Generated %d feature rows covering %d unique apps and %d runs.",
        len(feature_df),
        feature_df["track_id"].nunique(),
        feature_df["run_id"].nunique(),
    )


if __name__ == "__main__":
    main()
