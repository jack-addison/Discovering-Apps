#!/usr/bin/env python3
"""Neon/PostgreSQL variant of the Stage 1 snapshot scraper."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg
import requests

from src.local.stage1.app_store_scraper_v2 import (
    DEFAULT_HEADERS,
    ITUNES_CHART_PATHS,
    bounded_limit,
    fetch_category_map,
    fetch_search_results,
    lookup_apps,
    normalize_snapshot,
)
from src.prototype.config import load_settings


TARGET_LIMIT_PER_CATEGORY = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Apple App Store metadata and write snapshots to Neon/PostgreSQL."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--search-term", help="Keyword search for apps.")
    source_group.add_argument(
        "--collection",
        choices=["top-free", "top-paid", "top-grossing"],
        help="Chart collection to scrape.",
    )
    parser.add_argument(
        "--country",
        default="us",
        help="Two-letter country code (default: us).",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Iterate every category when scraping a collection.",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=100,
        help="Batch size for metadata lookup requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TARGET_LIMIT_PER_CATEGORY,
        help="Maximum apps per chart/category (default: 400).",
    )
    parser.add_argument(
        "--note",
        help="Optional note stored with the scrape run.",
    )
    parser.add_argument(
        "--postgres-dsn",
        help="Override PROTOTYPE_DATABASE_URL for the Neon connection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def record_scrape_run(
    conn: psycopg.Connection,
    *,
    source: str,
    country: str,
    collection: Optional[str],
    search_term: Optional[str],
    limit_requested: int,
    all_categories: bool,
    note: Optional[str],
) -> int:
    created_at = datetime.now(tz=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO scrape_runs (
                created_at, source, country, collection, search_term,
                limit_requested, all_categories, note
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                created_at,
                source,
                country,
                collection,
                search_term,
                limit_requested,
                all_categories,
                note,
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    logging.info("Created scrape run %s (%s)", run_id, created_at.isoformat())
    return run_id


def fetch_chart_page(
    collection: str,
    country: str,
    *,
    limit: int,
    genre_id: Optional[str] = None,
    offset: int = 0,
) -> List[int]:
    chart_path = ITUNES_CHART_PATHS.get(collection)
    if not chart_path:
        raise ValueError(f"Unsupported collection '{collection}'.")

    base_url = f"https://itunes.apple.com/{country}/rss/{chart_path}"
    if genre_id:
        base_url += f"/genre={genre_id}"
    base_url += f"/limit={bounded_limit(limit)}/json"

    params = {"offset": offset} if offset else None

    logging.debug("Requesting chart page %s (offset=%s)", base_url, offset)
    resp = requests.get(base_url, headers=DEFAULT_HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    feed = resp.json().get("feed", {})
    entries = feed.get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]

    ids: List[int] = []
    for entry in entries:
        raw_id = (
            entry.get("id", {}).get("attributes", {}).get("im:id")
            or entry.get("id", {}).get("label")
        )
        if not raw_id:
            continue
        try:
            ids.append(int(raw_id))
        except ValueError:
            logging.debug("Skipping non-numeric app id '%s'", raw_id)
    return ids


def fetch_extended_chart_ids(
    collection: str,
    country: str,
    limit: int,
    *,
    genre_id: Optional[str] = None,
) -> List[int]:
    results: List[int] = []
    seen: set[int] = set()
    offset = 0
    while len(results) < limit:
        batch_limit = min(200, limit - len(results))
        ids = fetch_chart_page(
            collection,
            country,
            limit=batch_limit,
            genre_id=genre_id,
            offset=offset,
        )
        if not ids:
            break
        new_ids = [track_id for track_id in ids if track_id not in seen]
        results.extend(new_ids)
        seen.update(new_ids)
        if len(ids) < batch_limit:
            break
        offset += len(ids)
    return results[:limit]


def build_category_memberships_extended(
    collection: str,
    country: str,
    limit: int,
) -> Tuple[Dict[int, List[Dict[str, object]]], List[Dict[str, object]]]:
    categories = fetch_category_map(country)
    membership_map: Dict[int, List[Dict[str, object]]] = {}
    ranking_rows: List[Dict[str, object]] = []
    for genre_id, name in sorted(categories.items(), key=lambda item: str(item[1]).lower()):
        ids = fetch_extended_chart_ids(collection, country, limit, genre_id=genre_id)
        if not ids:
            continue
        for rank, track_id in enumerate(ids, start=1):
            membership = {
                "chart_type": collection,
                "category_id": genre_id,
                "category_name": name,
                "rank": rank,
            }
            membership_map.setdefault(track_id, []).append(membership)
            ranking_rows.append(
                {
                    "track_id": track_id,
                    "chart_type": collection,
                    "category_id": genre_id,
                    "category_name": name,
                    "rank": rank,
                }
            )
    logging.info("Prepared memberships for %d categories.", len(membership_map))
    return membership_map, ranking_rows


def upsert_snapshots(
    conn: psycopg.Connection,
    run_id: int,
    records: Sequence[Dict],
    membership_map: Dict[int, List[Dict[str, object]]],
    scraped_at: str,
) -> None:
    if not records:
        logging.info("No snapshots to upsert.")
        return

    columns = [
        "run_id",
        "track_id",
        "name",
        "description",
        "release_date",
        "current_version_release_date",
        "version",
        "primary_genre_id",
        "primary_genre_name",
        "genre_ids",
        "genres",
        "content_advisory_rating",
        "price",
        "formatted_price",
        "currency",
        "is_free",
        "has_in_app_purchases",
        "seller_name",
        "seller_url",
        "developer_id",
        "bundle_id",
        "average_user_rating",
        "average_user_rating_current",
        "user_rating_count",
        "user_rating_count_current",
        "rating_count_list",
        "language_codes",
        "minimum_os_version",
        "file_size_bytes",
        "screenshot_urls",
        "ipad_screenshot_urls",
        "appletv_screenshot_urls",
        "app_store_url",
        "artwork_url",
        "chart_memberships",
        "scraped_at",
    ]

    payload: List[Tuple[object, ...]] = []
    for record in records:
        track_id = record.get("trackId")
        if track_id is None:
            continue
        memberships = membership_map.get(int(track_id), [])
        normalized = normalize_snapshot(
            record,
            run_id=run_id,
            memberships=memberships,
            scraped_at=scraped_at,
        )
        row: List[object] = []
        for col in columns:
            value = normalized.get(col)
            if col in {"is_free", "has_in_app_purchases"} and value is not None:
                value = bool(value)
            row.append(value)
        payload.append(tuple(row))

    if not payload:
        logging.info("No valid snapshot rows to insert after filtering.")
        return

    insert_sql = f"""
        INSERT INTO app_snapshots ({", ".join(columns)})
        VALUES ({", ".join(["%s"] * len(columns))})
        ON CONFLICT (run_id, track_id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            release_date = EXCLUDED.release_date,
            current_version_release_date = EXCLUDED.current_version_release_date,
            version = EXCLUDED.version,
            primary_genre_id = EXCLUDED.primary_genre_id,
            primary_genre_name = EXCLUDED.primary_genre_name,
            genre_ids = EXCLUDED.genre_ids,
            genres = EXCLUDED.genres,
            content_advisory_rating = EXCLUDED.content_advisory_rating,
            price = EXCLUDED.price,
            formatted_price = EXCLUDED.formatted_price,
            currency = EXCLUDED.currency,
            is_free = EXCLUDED.is_free,
            has_in_app_purchases = EXCLUDED.has_in_app_purchases,
            seller_name = EXCLUDED.seller_name,
            seller_url = EXCLUDED.seller_url,
            developer_id = EXCLUDED.developer_id,
            bundle_id = EXCLUDED.bundle_id,
            average_user_rating = EXCLUDED.average_user_rating,
            average_user_rating_current = EXCLUDED.average_user_rating_current,
            user_rating_count = EXCLUDED.user_rating_count,
            user_rating_count_current = EXCLUDED.user_rating_count_current,
            rating_count_list = EXCLUDED.rating_count_list,
            language_codes = EXCLUDED.language_codes,
            minimum_os_version = EXCLUDED.minimum_os_version,
            file_size_bytes = EXCLUDED.file_size_bytes,
            screenshot_urls = EXCLUDED.screenshot_urls,
            ipad_screenshot_urls = EXCLUDED.ipad_screenshot_urls,
            appletv_screenshot_urls = EXCLUDED.appletv_screenshot_urls,
            app_store_url = EXCLUDED.app_store_url,
            artwork_url = EXCLUDED.artwork_url,
            chart_memberships = EXCLUDED.chart_memberships,
            scraped_at = EXCLUDED.scraped_at;
    """

    with conn.cursor() as cur:
        cur.executemany(insert_sql, payload)
    conn.commit()
    logging.info("Upserted %d snapshots into Neon.", len(payload))


def upsert_rankings(
    conn: psycopg.Connection,
    run_id: int,
    ranking_rows: Sequence[Dict[str, object]],
) -> None:
    if not ranking_rows:
        return
    payload = [
        (
            run_id,
            row["track_id"],
            row["chart_type"],
            row["category_id"],
            row["category_name"],
            row["rank"],
        )
        for row in ranking_rows
    ]
    insert_sql = """
        INSERT INTO app_rankings (
            run_id, track_id, chart_type, category_id, category_name, rank
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, track_id, chart_type, category_id) DO UPDATE SET
            category_name = EXCLUDED.category_name,
            rank = EXCLUDED.rank;
    """
    with conn.cursor() as cur:
        cur.executemany(insert_sql, payload)
    conn.commit()
    logging.info("Upserted %d ranking rows.", len(payload))


def collect_raw_records(
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict[int, List[Dict[str, object]]], List[Dict[str, object]], str, str]:
    scraped_at = datetime.now(tz=timezone.utc).isoformat()
    if args.search_term:
        raw_records = fetch_search_results(args.search_term, args.country, args.limit)
        membership_map: Dict[int, List[Dict[str, object]]] = {}
        ranking_rows: List[Dict[str, object]] = []
        source = "search"
    elif args.all_categories:
        membership_map, ranking_rows = build_category_memberships_extended(
            args.collection,
            args.country,
            args.limit,
        )
        raw_records = lookup_apps(
            list(membership_map.keys()),
            args.country,
            args.lookup_batch_size,
        )
        source = "collection-all-categories"
    else:
        ids = fetch_extended_chart_ids(args.collection, args.country, args.limit)
        membership_map = {
            track_id: [
                {
                    "chart_type": args.collection,
                    "category_id": None,
                    "category_name": args.collection,
                    "rank": rank,
                }
            ]
            for rank, track_id in enumerate(ids, start=1)
        }
        ranking_rows = [
            {
                "track_id": track_id,
                "chart_type": args.collection,
                "category_id": None,
                "category_name": args.collection,
                "rank": rank,
            }
            for rank, track_id in enumerate(ids, start=1)
        ]
        raw_records = lookup_apps(ids, args.country, args.lookup_batch_size)
        source = "collection"

    logging.info("Retrieved %d detailed records.", len(raw_records))
    return raw_records, membership_map, ranking_rows, source, scraped_at


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    raw_records, membership_map, ranking_rows, source, scraped_at = collect_raw_records(args)
    if not raw_records:
        logging.warning("No metadata retrieved; aborting.")
        return

    settings = load_settings()
    postgres_dsn = args.postgres_dsn or settings.postgres_dsn

    with psycopg.connect(postgres_dsn) as conn:
        run_id = record_scrape_run(
            conn,
            source="app_store_scraper_neon",
            country=args.country,
            collection=args.collection,
            search_term=args.search_term,
            limit_requested=args.limit,
            all_categories=args.all_categories,
            note=args.note,
        )
        upsert_snapshots(conn, run_id, raw_records, membership_map, scraped_at)
        upsert_rankings(conn, run_id, ranking_rows)

    logging.info("Scrape complete (run id %s).", run_id)


if __name__ == "__main__":
    main()
