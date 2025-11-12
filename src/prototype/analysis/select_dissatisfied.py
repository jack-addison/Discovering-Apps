#!/usr/bin/env python3
"""Identify categories with high-volume but low-rated apps and store them."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import List, Sequence

import psycopg

from ..config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag high-volume poorly rated apps and persist them in app_snapshot_dissatisfied."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        action="append",
        help="Restrict processing to specific run IDs (repeatable).",
    )
    parser.add_argument(
        "--rating-quantile",
        type=float,
        default=0.7,
        help="Percentile of rating volume that defines 'high volume' within a category (default: 0.7).",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=3.0,
        help="Maximum average rating to be considered dissatisfied (default: 3.0).",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=50,
        help="Ignore apps with fewer than this many ratings (default: 50).",
    )
    parser.add_argument(
        "--postgres-dsn",
        help="Override PROTOTYPE_DATABASE_URL when connecting to Neon.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()
    if not 0 < args.rating_quantile < 1:
        parser.error("--rating-quantile must be between 0 and 1 (exclusive).")
    return args


def fetch_run_ids(conn: psycopg.Connection, requested: Sequence[int] | None) -> List[int]:
    if requested:
        return sorted(set(requested))
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT run_id FROM app_snapshots ORDER BY run_id")
        return [row[0] for row in cur.fetchall()]


def process_run(
    conn: psycopg.Connection,
    *,
    run_id: int,
    quantile: float,
    rating_threshold: float,
    min_ratings: int,
) -> int:
    logging.info("Processing run %s", run_id)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM app_snapshot_dissatisfied WHERE run_id = %s", (run_id,))
        cur.execute(
            """
            WITH ranked AS (
                SELECT
                    s.run_id,
                    s.track_id,
                    s.primary_genre_name AS category,
                    s.price,
                    s.average_user_rating,
                    s.user_rating_count,
                    PERCENT_RANK() OVER (
                        PARTITION BY s.primary_genre_name
                        ORDER BY s.user_rating_count
                    ) AS volume_percentile
                FROM app_snapshots AS s
                WHERE s.run_id = %s
                  AND s.average_user_rating IS NOT NULL
                  AND s.user_rating_count IS NOT NULL
                  AND s.user_rating_count >= %s
            )
            INSERT INTO app_snapshot_dissatisfied (
                run_id,
                track_id,
                category,
                price,
                average_user_rating,
                user_rating_count,
                rating_percentile,
                threshold_rating,
                flagged_at
            )
            SELECT
                ranked.run_id,
                ranked.track_id,
                ranked.category,
                ranked.price,
                ranked.average_user_rating,
                ranked.user_rating_count,
                ranked.volume_percentile,
                %s AS threshold_rating,
                NOW()
            FROM ranked
            WHERE ranked.average_user_rating < %s
              AND ranked.volume_percentile >= %s
            ON CONFLICT (run_id, track_id) DO UPDATE SET
                category = EXCLUDED.category,
                price = EXCLUDED.price,
                average_user_rating = EXCLUDED.average_user_rating,
                user_rating_count = EXCLUDED.user_rating_count,
                rating_percentile = EXCLUDED.rating_percentile,
                threshold_rating = EXCLUDED.threshold_rating,
                flagged_at = EXCLUDED.flagged_at
            RETURNING 1
            """,
            (run_id, min_ratings, rating_threshold, rating_threshold, quantile),
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    dsn = args.postgres_dsn or settings.postgres_dsn

    with psycopg.connect(dsn) as conn:
        run_ids = fetch_run_ids(conn, args.run_id)
        total_flagged = 0
        for run_id in run_ids:
            total_flagged += process_run(
                conn,
                run_id=run_id,
                quantile=args.rating_quantile,
                rating_threshold=args.rating_threshold,
                min_ratings=args.min_ratings,
            )
        logging.info("Flagged %s dissatisfied snapshots across %s runs.", total_flagged, len(run_ids))


if __name__ == "__main__":
    main()
