#!/usr/bin/env python3
"""Reuse Stage 2 scores for unchanged snapshots in SQLiteCloud."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import sqlitecloud
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_config import CONNECTION_URI

MAX_RATING_DELTA_ABS = 50
MAX_RATING_DELTA_PCT = 0.05
MAX_RANK_DELTA = 20


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
    parser = argparse.ArgumentParser(description="Reuse Stage 2 scores in SQLiteCloud when metadata hasn't changed.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def fetch_dataframe(conn, query: str) -> List[Dict]:
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    return [dict(zip(columns, row)) for row in rows]


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


def load_scored_snapshots(conn) -> Dict[int, Snapshot]:
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
    lookup: Dict[int, Snapshot] = {}
    for row in fetch_dataframe(conn, query):
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


def load_pending_snapshots(conn) -> List[Snapshot]:
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
    snapshots: List[Snapshot] = []
    for row in fetch_dataframe(conn, query):
        snapshots.append(
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
        )
    return snapshots


def should_reuse(current: Snapshot, previous: Snapshot) -> bool:
    if current.description != previous.description:
        return False
    if current.version != previous.version:
        return False
    if current.price != previous.price or current.currency != previous.currency:
        return False

    if current.user_rating_count is not None and previous.user_rating_count is not None:
        delta = abs(current.user_rating_count - previous.user_rating_count)
        if delta > MAX_RATING_DELTA_ABS and delta > previous.user_rating_count * MAX_RATING_DELTA_PCT:
            return False

    if current.average_user_rating is not None and previous.average_user_rating is not None:
        if abs(current.average_user_rating - previous.average_user_rating) > 0.1:
            return False

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


def update_snapshot(conn, current: Snapshot, previous: Snapshot) -> None:
    conn.execute(
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

    with sqlitecloud.connect(args.connection_uri) as conn:
        scored_lookup = load_scored_snapshots(conn)
        pending = load_pending_snapshots(conn)
        reused = 0
        for snapshot in pending:
            previous = scored_lookup.get(snapshot.track_id)
            if not previous:
                continue
            if should_reuse(snapshot, previous):
                update_snapshot(conn, snapshot, previous)
                reused += 1
        if reused:
            conn.commit()
        logging.info("Reuse complete. Reused %d snapshots.", reused)


if __name__ == "__main__":
    main()
