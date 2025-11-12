#!/usr/bin/env python3
"""Experimental script to batch Stage 2 scoring for comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import APIError, OpenAI, RateLimitError

DEFAULT_DB_PATH = Path("exports") / "app_store_apps_v2.db"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BATCH_SIZE = 20
DEFAULT_LIMIT = 40

SYSTEM_PROMPT = (
    "You are a seasoned mobile product strategist. "
    "Estimate build effort and success likelihood for each provided iOS app. "
    "Respond in valid JSON that matches the supplied schema."
)

USER_TEMPLATE = """You will receive a JSON array of app metadata.\n\nReturn a JSON object with a single key `results` whose value is a list of objects. Each object MUST contain:\n- `track_id` (integer copied from the input)\n- `build_time_weeks` (float)\n- `success_score` (float in [0, 100])\n- `reasoning` (string, <= 300 characters, concise justification).\n\nProcess each app independently—do not let information leak between entries. Preserve the order of the input array."""

@dataclass
class Snapshot:
    run_id: int
    track_id: int
    name: str
    category: str | None
    review_score: float | None
    number_of_ratings: int | None
    number_of_downloads: int | None
    price: float | None
    currency: str | None
    developer: str | None
    description: str
    language_codes: str | None
    chart_memberships: str | None
    existing_build_time: float | None
    existing_success_score: float | None
    existing_success_reasoning: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Stage 2 scoring experiment. Fetches the first N snapshots, sends"
            " batched prompts to OpenAI, and stores the responses for comparison"
            " without writing to the database."
        )
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
        help="Number of snapshots to sample for the experiment (default: 40).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of apps to include per OpenAI request (default: 20).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "experiments" / "batch_stage2_results.json",
        help="Destination JSON file for experiment output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")


def fetch_snapshots(
    connection: sqlite3.Connection,
    *,
    limit: int,
) -> List[Snapshot]:
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


def summarize_memberships(chart_memberships: str | None) -> List[Dict[str, Any]]:
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


def build_app_payload(snapshot: Snapshot) -> Dict[str, Any]:
    memberships = summarize_memberships(snapshot.chart_memberships)
    top_memberships = sorted(
        memberships,
        key=lambda item: item.get("rank", float("inf")),
    )[:3]
    return {
        "track_id": snapshot.track_id,
        "run_id": snapshot.run_id,
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


def chunked(iterable: Sequence[Snapshot], size: int) -> Iterable[List[Snapshot]]:
    chunk: List[Snapshot] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def call_openai_batch(
    client: OpenAI,
    model: str,
    batch_payload: List[Dict[str, Any]],
    *,
    max_retries: int = 5,
    retry_wait: int = 5,
) -> Dict[int, Dict[str, Any]]:
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
                        "content": USER_TEMPLATE + "\n\nApps:\n" + json.dumps(batch_payload, ensure_ascii=False, indent=2),
                    },
                ],
                temperature=0.2,
            )
            payload = json.loads(response.output_text)
            results = payload.get("results", [])
            return {int(item["track_id"]): item for item in results}
        except (RateLimitError, APIError) as err:
            if attempt >= max_retries:
                raise
            logging.warning(
                "OpenAI batch call failed (attempt %d/%d): %s. Retrying in %s seconds.",
                attempt,
                max_retries,
                err,
                retry_wait,
            )
            time.sleep(retry_wait)
        except json.JSONDecodeError as err:
            logging.error("Failed to decode batch response: %s", err)
            raise


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    ensure_api_key()

    connection = sqlite3.connect(args.db_path)
    snapshots = fetch_snapshots(connection, limit=args.limit)
    connection.close()

    if not snapshots:
        logging.warning("No snapshots found for the experiment.")
        return

    logging.info("Loaded %d snapshots for experiment (batch size %d).", len(snapshots), args.batch_size)
    client = OpenAI()
    experiment_rows: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(chunked(snapshots, args.batch_size), start=1):
        logging.info("Processing batch %d with %d apps", batch_idx, len(batch))
        batch_payload = [build_app_payload(snapshot) for snapshot in batch]
        results = call_openai_batch(client, args.model, batch_payload)

        for snapshot in batch:
            result = results.get(snapshot.track_id)
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
                    "batch_result": result,
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
                    "batch_size": args.batch_size,
                },
                "results": experiment_rows,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Experiment complete. Results written to %s", args.output)


if __name__ == "__main__":
    main()
