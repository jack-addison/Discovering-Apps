#!/usr/bin/env python3
"""Compute snapshot deltas against the Neon/PostgreSQL database."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

import pandas as pd
import psycopg

from src.local.analysis.build_deltas import (
    QUERY,
    _parse_best_rank,
    compute_deltas,
)
from ..config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize app_snapshot_deltas inside Neon.")
    parser.add_argument("--postgres-dsn", help="Override PROTOTYPE_DATABASE_URL for Neon.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--recreate-table", action="store_true")
    return parser.parse_args()


def load_dataframe(dsn: str) -> pd.DataFrame:
    with psycopg.connect(dsn) as conn:
        df = pd.read_sql_query(QUERY, conn)
    if df.empty:
        raise ValueError("No snapshot data returned from database.")
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    df.sort_values(["track_id", "run_id"], inplace=True)
    df["best_rank"] = df["chart_memberships"].apply(_parse_best_rank).astype("float64")
    return df


def ensure_table(conn: psycopg.Connection, recreate: bool) -> None:
    with conn.cursor() as cur:
        if recreate:
            cur.execute("DROP TABLE IF EXISTS app_snapshot_deltas")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_snapshot_deltas (
                run_id INTEGER NOT NULL,
                run_created_at TIMESTAMPTZ,
                track_id BIGINT NOT NULL,
                name TEXT,
                category TEXT,
                price DOUBLE PRECISION,
                currency TEXT,
                average_user_rating DOUBLE PRECISION,
                user_rating_count DOUBLE PRECISION,
                build_time_estimate DOUBLE PRECISION,
                success_score DOUBLE PRECISION,
                success_reasoning TEXT,
                best_rank DOUBLE PRECISION,
                description TEXT,
                version TEXT,
                developer TEXT,
                prev_run_id DOUBLE PRECISION,
                prev_run_created_at TIMESTAMPTZ,
                prev_success_score DOUBLE PRECISION,
                prev_build_time DOUBLE PRECISION,
                prev_rating DOUBLE PRECISION,
                prev_rating_count DOUBLE PRECISION,
                prev_price DOUBLE PRECISION,
                prev_currency TEXT,
                prev_rank DOUBLE PRECISION,
                is_new_track BOOLEAN,
                delta_success DOUBLE PRECISION,
                delta_build_time DOUBLE PRECISION,
                delta_rating DOUBLE PRECISION,
                delta_rating_count DOUBLE PRECISION,
                delta_price DOUBLE PRECISION,
                delta_rank DOUBLE PRECISION,
                price_changed BOOLEAN,
                days_since_prev DOUBLE PRECISION,
                PRIMARY KEY (run_id, track_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_track ON app_snapshot_deltas(track_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_app_deltas_run ON app_snapshot_deltas(run_id)")
    conn.commit()


def write_output(conn: psycopg.Connection, df: pd.DataFrame) -> None:
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
    working = df[columns].copy()
    for col in ["run_created_at", "prev_run_created_at"]:
        if col in working.columns:
            working[col] = working[col].apply(lambda x: x.tz_localize(None) if isinstance(x, pd.Timestamp) and x.tzinfo else x)
            working[col] = working[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    for bool_col in ["is_new_track", "price_changed"]:
        if bool_col in working.columns:
            working[bool_col] = working[bool_col].apply(
                lambda val: bool(val) if pd.notna(val) else None
            )
    for int_col in ["run_id", "track_id"]:
        if int_col in working.columns:
            working[int_col] = working[int_col].apply(
                lambda val: int(val) if pd.notna(val) else None
            )
    for float_col in ["user_rating_count", "prev_rating_count", "delta_rating_count", "prev_run_id"]:
        if float_col in working.columns:
            working[float_col] = working[float_col].apply(
                lambda val: float(val) if pd.notna(val) else None
            )
    working = working.replace({pd.NA: None})
    records = working.to_dict("records")
    insert_sql = """
        INSERT INTO app_snapshot_deltas (
            run_id, run_created_at, track_id, name, category, price, currency,
            average_user_rating, user_rating_count, build_time_estimate, success_score,
            success_reasoning, best_rank, description, version, developer,
            prev_run_id, prev_run_created_at, prev_success_score, prev_build_time,
            prev_rating, prev_rating_count, prev_price, prev_currency, prev_rank,
            is_new_track, delta_success, delta_build_time, delta_rating,
            delta_rating_count, delta_price, delta_rank, price_changed, days_since_prev
        ) VALUES (
            %(run_id)s, %(run_created_at)s, %(track_id)s, %(name)s, %(category)s, %(price)s, %(currency)s,
            %(average_user_rating)s, %(user_rating_count)s, %(build_time_estimate)s, %(success_score)s,
            %(success_reasoning)s, %(best_rank)s, %(description)s, %(version)s, %(developer)s,
            %(prev_run_id)s, %(prev_run_created_at)s, %(prev_success_score)s, %(prev_build_time)s,
            %(prev_rating)s, %(prev_rating_count)s, %(prev_price)s, %(prev_currency)s, %(prev_rank)s,
            %(is_new_track)s, %(delta_success)s, %(delta_build_time)s, %(delta_rating)s,
            %(delta_rating_count)s, %(delta_price)s, %(delta_rank)s, %(price_changed)s, %(days_since_prev)s
        )
        ON CONFLICT (run_id, track_id) DO UPDATE SET
            run_created_at = EXCLUDED.run_created_at,
            name = EXCLUDED.name,
            category = EXCLUDED.category,
            price = EXCLUDED.price,
            currency = EXCLUDED.currency,
            average_user_rating = EXCLUDED.average_user_rating,
            user_rating_count = EXCLUDED.user_rating_count,
            build_time_estimate = EXCLUDED.build_time_estimate,
            success_score = EXCLUDED.success_score,
            success_reasoning = EXCLUDED.success_reasoning,
            best_rank = EXCLUDED.best_rank,
            description = EXCLUDED.description,
            version = EXCLUDED.version,
            developer = EXCLUDED.developer,
            prev_run_id = EXCLUDED.prev_run_id,
            prev_run_created_at = EXCLUDED.prev_run_created_at,
            prev_success_score = EXCLUDED.prev_success_score,
            prev_build_time = EXCLUDED.prev_build_time,
            prev_rating = EXCLUDED.prev_rating,
            prev_rating_count = EXCLUDED.prev_rating_count,
            prev_price = EXCLUDED.prev_price,
            prev_currency = EXCLUDED.prev_currency,
            prev_rank = EXCLUDED.prev_rank,
            is_new_track = EXCLUDED.is_new_track,
            delta_success = EXCLUDED.delta_success,
            delta_build_time = EXCLUDED.delta_build_time,
            delta_rating = EXCLUDED.delta_rating,
            delta_rating_count = EXCLUDED.delta_rating_count,
            delta_price = EXCLUDED.delta_price,
            delta_rank = EXCLUDED.delta_rank,
            price_changed = EXCLUDED.price_changed,
            days_since_prev = EXCLUDED.days_since_prev
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM app_snapshot_deltas")
        chunk = 500
        for start in range(0, len(records), chunk):
            batch = records[start : start + chunk]
            try:
                cur.executemany(insert_sql, batch)
            except psycopg.errors.NumericValueOutOfRange as exc:
                diag = getattr(exc, "diag", None)
                column = getattr(diag, "column_name", None) if diag else None
                offending = batch[:1]
                logging.error("Numeric overflow while inserting deltas (column=%s, sample=%s)", column, offending)
                raise
        conn.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    settings = load_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn

    df = load_dataframe(dsn)
    deltas = compute_deltas(df)
    with psycopg.connect(dsn) as conn:
        ensure_table(conn, args.recreate_table)
        write_output(conn, deltas)


if __name__ == "__main__":
    main()
