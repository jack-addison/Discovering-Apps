#!/usr/bin/env python3
"""Evaluate cheaper OpenAI models against existing Stage 2 scores."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import APIError, OpenAI, RateLimitError

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"
DEFAULT_MODEL = "gpt-3.5-turbo"  # swap for desired cheaper model
DEFAULT_LIMIT = 40

SYSTEM_PROMPT = (
    "You are a seasoned mobile product strategist. "
    "Estimate build effort and success likelihood for the provided iOS app. "
    "Respond with a single JSON object and no additional commentary."
)

USER_PROMPT_TEMPLATE = """Using the following app metadata, produce a JSON object matching this schema:\n{{\n  \"build_time_weeks\": <float>,\n  \"success_score\": <float>,\n  \"reasoning\": <string>\n}}\n\nGuidelines:\n- Consider feature complexity, integrations, content requirements, and UI depth when estimating build time.\n- Use review volume, rating average, chart rank, developer reputation, and category competitiveness for success score.\n- Clamp success_score to [0, 100].\n- If information is missing, make conservative assumptions and mention them in reasoning.\n- Return strictly valid JSON without comments or trailing text.\n\nApp metadata:\n{payload}\n"""


@dataclass
class Snapshot:
    run_id: int
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
    existing_success_reasoning: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a cheaper OpenAI model against existing Stage 2 scores without modifying the database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the snapshot database (default: exports/app_store_apps_v2.db).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of snapshots to sample (default: 40).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Cheaper OpenAI model to evaluate (default: gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments") / "cheaper_model_results.json",
        help="Where to write the comparison JSON output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per API call when rate limited or on transient errors.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=5,
        help="Seconds to wait between retries (default: 5).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")


def fetch_snapshots(connection: sqlite3.Connection, limit: int) -> List[Snapshot]:
    query = """
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
        WHERE build_time_estimate IS NOT NULL
          AND success_score IS NOT NULL
        ORDER BY run_id DESC, track_id
        LIMIT ?
    """
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, (limit,)).fetchall()
    return [
        Snapshot(
            run_id=row["run_id"],
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
            existing_success_reasoning=row["success_reasoning"],
        )
        for row in rows
    ]


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


def build_prompt_payload(snapshot: Snapshot) -> Dict[str, Any]:
    memberships = summarize_memberships(snapshot.chart_memberships)
    top_memberships = sorted(
        memberships,
        key=lambda item: item.get("rank", float("inf")),
    )[:3]
    return {
        "run_id": snapshot.run_id,
        "track_id": snapshot.track_id,
        "app": {
            "name": snapshot.name,
            "category": snapshot.category,
            "developer": snapshot.developer,
            "price": snapshot.price,
            "currency": snapshot.currency,
            "review_score": snapshot.review_score,
            "number_of_ratings": snapshot.number_of_ratings,
            "downloads_proxy": snapshot.number_of_downloads,
            "language_codes": snapshot.language_codes,
            "top_chart_memberships": top_memberships,
            "description": snapshot.description[:4000],
        },
    }


def call_openai_single(
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            payload=json.dumps(payload, ensure_ascii=False, indent=2)
                        ),
                    },
                ],
                temperature=0.2,
            )
            return json.loads(response.output_text)
        except (RateLimitError, APIError) as err:
            if attempt >= max_retries:
                raise
            logging.warning(
                "API call failed (attempt %d/%d): %s. Retrying in %s seconds.",
                attempt,
                max_retries,
                err,
                retry_wait,
            )
            time.sleep(retry_wait)
        except json.JSONDecodeError as err:
            logging.error("Failed to decode JSON response: %s", err)
            raise


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    ensure_api_key()

    connection = sqlite3.connect(args.db_path)
    snapshots = fetch_snapshots(connection, limit=args.limit)
    connection.close()

    if not snapshots:
        logging.warning("No snapshots found; aborting experiment.")
        return

    logging.info("Loaded %d snapshots for comparison using %s", len(snapshots), args.model)
    client = OpenAI()
    experiment_rows: List[Dict[str, Any]] = []

    for snapshot in snapshots:
        payload = build_prompt_payload(snapshot)
        try:
            result = call_openai_single(
                client,
                args.model,
                payload,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
        except Exception as err:  # noqa: BLE001
            logging.error(
                "Failed to score track_id %s (%s) with %s: %s",
                snapshot.track_id,
                snapshot.name,
                args.model,
                err,
            )
            result = None

        experiment_rows.append(
            {
                "run_id": snapshot.run_id,
                "track_id": snapshot.track_id,
                "name": snapshot.name,
                "existing": {
                    "build_time_weeks": snapshot.existing_build_time,
                    "success_score": snapshot.existing_success_score,
                    "reasoning": snapshot.existing_success_reasoning,
                },
                "cheaper_model_result": result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "metadata": {
                    "db_path": str(args.db_path),
                    "model": args.model,
                    "limit": args.limit,
                },
                "results": experiment_rows,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Experiment complete. Results stored in %s", args.output)


if __name__ == "__main__":
    main()

