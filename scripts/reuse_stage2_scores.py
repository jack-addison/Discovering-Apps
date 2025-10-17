#!/usr/bin/env python3
"""Copy Stage 2 scores to new snapshots when metadata has not meaningfully changed."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"

# thresholds for reuse decisions
MAX_RATING_DELTA_ABS = 50     # allow up to 50 extra ratings without re-score
MAX_RATING_DELTA_PCT = 0.05   # or 5% growth
MAX_RANK_DELTA = 20           # tolerate small chart rank shifts
MAX_BUILD_TIME_AGE_DAYS = 90  # re-score if older than ~3 months (optional)


@dataclass
class Snapshot:
    run_id: int
    track_id: int
    description: str
    version: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    average_user_rating: Optional[float]
    user_rating_count: Optional[int]
    chart_memberships: List[Dict]
    scraped_at: Optional[str]
    build_time_estimate: Optional[float]
    success_score: Optional[float]
    success_reasoning: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse Stage 2 scores when app metadata is unchanged within tolerances."
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
    return parser.parse_args()


def load_scored_snapshots(connection: sqlite3.Connection) -> Dict[int, Snapshot]:
    query = """
        SELECT
            run_id,
            track_id,
            description,
            version,
            price,
            currency,
            average_user_rating,
            user_rating_count,
            chart_memberships,
            scraped_at,
            build_time_estimate,
            success_score,
            success_reasoning
        FROM app_snapshots
        WHERE build_time_estimate IS NOT NULL
          AND success_score IS NOT NULL
          AND success_reasoning IS NOT NULL
        ORDER BY run_id DESC
    """
    connection.row_factory = sqlite3.Row
    lookup: Dict[int, Snapshot] = {}
    for row in connection.execute(query):
        track_id = row["track_id"]
        if track_id not in lookup:
            lookup[track_id] = Snapshot(
                run_id=row["run_id"],
                track_id=track_id,
                description=row["description"],
                version=row["version"],
                price=row["price"],
                currency=row["currency"],
                average_user_rating=row["average_user_rating"],
                user_rating_count=row["user_rating_count"],
                chart_memberships=parse_chart_memberships(row["chart_memberships"]),
                scraped_at=row["scraped_at"],
                build_time_estimate=row["build_time_estimate"],
                success_score=row["success_score"],
                success_reasoning=row["success_reasoning"],
            )
    return lookup


def parse_chart_memberships(raw: Optional[str]) -> List[Dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        logging.debug("Unable to parse chart memberships: %s", raw)
    return []


def load_pending_snapshots(connection: sqlite3.Connection) -> List[Snapshot]:
    query = """
        SELECT
            run_id,
            track_id,
            description,
            version,
            price,
            currency,
            average_user_rating,
            user_rating_count,
            chart_memberships,
            scraped_at,
            build_time_estimate,
            success_score,
            success_reasoning
        FROM app_snapshots
        WHERE build_time_estimate IS NULL
           OR success_score IS NULL
           OR success_reasoning IS NULL
        ORDER BY run_id, track_id
    """
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query).fetchall()
    return [
        Snapshot(
            run_id=row["run_id"],
            track_id=row["track_id"],
            description=row["description"],
            version=row["version"],
            price=row["price"],
            currency=row["currency"],
            average_user_rating=row["average_user_rating"],
            user_rating_count=row["user_rating_count"],
            chart_memberships=parse_chart_memberships(row["chart_memberships"]),
            scraped_at=row["scraped_at"],
            build_time_estimate=row["build_time_estimate"],
            success_score=row["success_score"],
            success_reasoning=row["success_reasoning"],
        )
        for row in rows
    ]


def should_reuse(current: Snapshot, previous: Snapshot) -> bool:
    # critical fields must match exactly
    if current.description != previous.description:
        return False
    if current.version != previous.version:
        return False
    if current.price != previous.price or current.currency != previous.currency:
        return False

    # rating volume tolerance
    if current.user_rating_count is not None and previous.user_rating_count is not None:
        delta = abs(current.user_rating_count - previous.user_rating_count)
        if delta > MAX_RATING_DELTA_ABS and delta > previous.user_rating_count * MAX_RATING_DELTA_PCT:
            return False

    # rating score tolerance (small rounding drift)
    if current.average_user_rating is not None and previous.average_user_rating is not None:
        if abs(current.average_user_rating - previous.average_user_rating) > 0.1:
            return False

    # chart membership tolerance (rank shifts)
    prev_ranks = {(entry.get("category_id"), entry.get("chart_type")): entry.get("rank") for entry in previous.chart_memberships}
    for entry in current.chart_memberships:
        key = (entry.get("category_id"), entry.get("chart_type"))
        prev_rank = prev_ranks.get(key)
        rank = entry.get("rank")
        if prev_rank is None or rank is None:
            continue
        if abs(rank - prev_rank) > MAX_RANK_DELTA:
            return False

    return True


def update_snapshot(connection: sqlite3.Connection, current: Snapshot, previous: Snapshot) -> None:
    connection.execute(
        """
        UPDATE app_snapshots
        SET build_time_estimate = ?, success_score = ?, success_reasoning = ?
        WHERE run_id = ? AND track_id = ?
        """,
        (
            previous.build_time_estimate,
            previous.success_score,
            previous.success_reasoning,
            current.run_id,
            current.track_id,
        ),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found at {args.db_path}")

    connection = sqlite3.connect(args.db_path)

    logging.info("Loading prior scored snapshots...")
    scored_lookup = load_scored_snapshots(connection)
    logging.info("Loaded %d previously-scored track IDs", len(scored_lookup))

    pending = load_pending_snapshots(connection)
    logging.info("Found %d pending snapshots needing scores", len(pending))

    reused = 0
    skipped = 0
    for snapshot in pending:
        previous = scored_lookup.get(snapshot.track_id)
        if not previous:
            skipped += 1
            continue
        if should_reuse(snapshot, previous):
            update_snapshot(connection, snapshot, previous)
            reused += 1
        else:
            skipped += 1

    connection.commit()
    connection.close()

    logging.info("Reuse complete. Reused %d snapshots, left %d for Stage 2.", reused, skipped)


if __name__ == "__main__":
    main()
