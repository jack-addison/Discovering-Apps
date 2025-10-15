#!/usr/bin/env python3
"""Augment the scraped App Store dataset with build effort and success estimates via OpenAI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import APIError, OpenAI, RateLimitError

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SLEEP_SECONDS = 5
SYSTEM_PROMPT = (
    "You are a seasoned mobile product strategist. "
    "Estimate build effort and success likelihood for the provided iOS app. "
    "Respond with a single JSON object and no additional commentary."
)
USER_PROMPT_TEMPLATE = """Using the following app metadata, produce a JSON object matching this schema:
{{
  "build_time_weeks": <float>,         // estimated weeks for a senior team to build an MVP
  "success_score": <float>,            // 0-100 likelihood of traction
  "reasoning": <string>                // concise justification (max ~300 characters)
}}

Guidelines:
- Consider feature complexity, integrations, content requirements, and UI depth when estimating build time.
- Use review volume, rating average, chart rank, developer reputation, and category competitiveness for success score.
- Clamp success_score to [0, 100].
- If data is missing, make conservative assumptions and mention them in reasoning.
- Return strictly valid JSON without comments or trailing text.

App metadata:
{payload}
"""


@dataclass
class AppRecord:
    track_id: int
    name: str
    category: Optional[str]
    review_score: Optional[float]
    number_of_ratings: Optional[int]
    number_of_downloads: Optional[int]
    price: Optional[float]
    currency: Optional[str]
    developer: Optional[str]
    description: str
    language_codes: Optional[str]
    chart_memberships: Optional[str]
    existing_build_time: Optional[float]
    existing_success_score: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate build_time_estimate and success_score columns using the OpenAI API."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("exports") / "app_store_apps.db",
        help="Path to the SQLite database created in Stage 1.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model id to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries per OpenAI call when rate limited or transient errors occur.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds to wait between retries (default: {DEFAULT_SLEEP_SECONDS}).",
    )
    parser.add_argument(
        "--batch-progress",
        type=int,
        default=20,
        help="Print progress every N processed apps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score apps even if build_time_estimate and success_score already exist.",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        help="Limit the number of apps to process (useful for smoke tests).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Environment variable OPENAI_API_KEY must be set before running this script.")

    return args


def ensure_columns(connection: sqlite3.Connection) -> None:
    """Add the Stage 2 columns if they are missing."""
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(apps)")
    }
    required_columns = {
        "build_time_estimate": "REAL",
        "success_score": "REAL",
    }
    for column, column_type in required_columns.items():
        if column not in existing_columns:
            logging.info("Adding missing column '%s' to apps table", column)
            connection.execute(f"ALTER TABLE apps ADD COLUMN {column} {column_type}")
    connection.commit()


def fetch_apps(
    connection: sqlite3.Connection,
    *,
    include_existing: bool,
    limit: Optional[int],
) -> List[AppRecord]:
    where_clause = ""
    params: Tuple[Any, ...] = ()
    if not include_existing:
        where_clause = "WHERE build_time_estimate IS NULL OR success_score IS NULL"
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params = params + (limit,)

    query = f"""
        SELECT
            track_id,
            name,
            category,
            review_score,
            number_of_ratings,
            number_of_downloads,
            price,
            currency,
            developer,
            description,
            language_codes,
            chart_memberships,
            build_time_estimate,
            success_score
        FROM apps
        {where_clause}
        ORDER BY track_id
        {limit_clause}
    """
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, params).fetchall()
    apps = [
        AppRecord(
            track_id=row["track_id"],
            name=row["name"],
            category=row["category"],
            review_score=row["review_score"],
            number_of_ratings=row["number_of_ratings"],
            number_of_downloads=row["number_of_downloads"],
            price=row["price"],
            currency=row["currency"],
            developer=row["developer"],
            description=row["description"] or "",
            language_codes=row["language_codes"],
            chart_memberships=row["chart_memberships"],
            existing_build_time=row["build_time_estimate"],
            existing_success_score=row["success_score"],
        )
        for row in rows
    ]
    return apps


def summarize_memberships(chart_memberships: Optional[str]) -> List[Dict[str, Any]]:
    if not chart_memberships:
        return []
    try:
        parsed = json.loads(chart_memberships)
    except json.JSONDecodeError:
        logging.debug("Failed to decode chart memberships: %s", chart_memberships)
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def build_prompt_payload(app: AppRecord) -> Dict[str, Any]:
    memberships = summarize_memberships(app.chart_memberships)
    top_memberships = sorted(
        memberships,
        key=lambda item: item.get("rank", float("inf")),
    )[:3]
    payload = {
        "app": {
            "name": app.name,
            "category": app.category,
            "developer": app.developer,
            "price": app.price,
            "currency": app.currency,
            "review_score": app.review_score,
            "number_of_ratings": app.number_of_ratings,
            "downloads_proxy": app.number_of_downloads,
            "language_codes": app.language_codes,
            "top_chart_memberships": top_memberships,
            "description": app.description[:4000],  # stay within token budget
        },
        "instructions": {
            "success_score_definition": (
                "Assign a 0-100 confidence score for commercial/environmental traction."
            ),
            "build_time_context": (
                "Estimate weeks needed for a small senior team to ship an MVP with core features."
            ),
            "assumptions": [
                "Assume cloud/back-end infrastructure is built from scratch.",
                "Weigh UI complexity, integrations, real-time features, and content requirements.",
                "Use download proxies, rating volume, rank, and developer reputation for market traction.",
                "If information is missing, infer cautiously and explain assumptions.",
            ],
        },
    }
    return payload


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
    *,
    max_retries: int,
    retry_wait: int,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            payload=json.dumps(payload, ensure_ascii=False, indent=2)
                        ),
                    },
                ],
                temperature=0.2,
            )
            text = response.output_text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as err:
                if attempt >= max_retries:
                    logging.error(
                        "Failed to parse JSON after %d attempts. Last response: %s",
                        attempt,
                        text,
                    )
                    raise
                logging.warning(
                    "Unable to parse JSON response (attempt %d/%d): %s. Retrying in %s seconds.",
                    attempt,
                    max_retries,
                    err,
                    retry_wait,
                )
                time.sleep(retry_wait)
        except (RateLimitError, APIError) as err:
            if attempt >= max_retries:
                raise
            logging.warning(
                "OpenAI call failed (attempt %d/%d): %s. Retrying in %s seconds.",
                attempt,
                max_retries,
                err,
                retry_wait,
            )
            time.sleep(retry_wait)
        except json.JSONDecodeError as err:
            logging.error("Failed to decode JSON response: %s", err)
            raise


def process_apps(
    connection: sqlite3.Connection,
    apps: Sequence[AppRecord],
    *,
    model: str,
    max_retries: int,
    retry_wait: int,
    batch_progress: int,
) -> Tuple[int, int]:
    client = OpenAI()
    updated = 0
    skipped = 0
    for idx, app in enumerate(apps, start=1):
        if app.existing_build_time is not None and app.existing_success_score is not None:
            logging.debug("Skipping track_id %s (already scored)", app.track_id)
            skipped += 1
            continue
        payload = build_prompt_payload(app)
        try:
            result = call_openai_with_retry(
                client,
                model,
                payload,
                max_retries=max_retries,
                retry_wait=retry_wait,
            )
        except Exception as err:  # noqa: BLE001
            logging.error(
                "Failed to score track_id %s (%s): %s",
                app.track_id,
                app.name,
                err,
            )
            skipped += 1
            continue

        build_time = result.get("build_time_weeks")
        success_score = result.get("success_score")
        reasoning = result.get("reasoning")

        if build_time is None or success_score is None:
            logging.error(
                "Incomplete response for track_id %s (%s): %s",
                app.track_id,
                app.name,
                result,
            )
            skipped += 1
            continue

        try:
            build_time = float(build_time)
            success_score = float(success_score)
        except (TypeError, ValueError):
            logging.error(
                "Non-numeric response for track_id %s (%s): %s",
                app.track_id,
                app.name,
                result,
            )
            skipped += 1
            continue

        success_score = max(0.0, min(100.0, success_score))

        logging.debug(
            "Scored track_id %s: build_time=%.2f weeks, success_score=%.2f. Reasoning: %s",
            app.track_id,
            build_time,
            success_score,
            reasoning,
        )

        connection.execute(
            """
            UPDATE apps
            SET build_time_estimate = ?, success_score = ?
            WHERE track_id = ?
            """,
            (build_time, success_score, app.track_id),
        )
        connection.commit()
        updated += 1

        if batch_progress > 0 and idx % batch_progress == 0:
            logging.info(
                "Processed %d/%d apps (updated=%d, skipped=%d)...",
                idx,
                len(apps),
                updated,
                skipped,
            )

    return updated, skipped


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database file not found: {args.db_path}")

    connection = sqlite3.connect(args.db_path)
    ensure_columns(connection)

    apps = fetch_apps(
        connection,
        include_existing=args.force,
        limit=args.max_apps,
    )

    if not apps:
        logging.info("No apps satisfied the selection criteria. Nothing to do.")
        return

    logging.info("Loaded %d apps for processing.", len(apps))
    updated, skipped = process_apps(
        connection,
        apps,
        model=args.model,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        batch_progress=args.batch_progress,
    )
    logging.info(
        "Finished Stage 2: updated %d app rows, skipped %d (already complete or errored).",
        updated,
        skipped,
    )
    connection.close()


if __name__ == "__main__":
    main()
