#!/usr/bin/env python3
"""Enhanced Apple App Store scraper that preserves historical snapshots."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

SEARCH_API_URL = "https://itunes.apple.com/search"
LOOKUP_API_URL = "https://itunes.apple.com/lookup"
RSS_COLLECTION_URL = (
    "https://rss.applemarketingtools.com/api/v2/{country}/apps/{collection}/{limit}/apps.json"
)
GENRE_TREE_URL = "https://itunes.apple.com/WebObjects/MZStoreServices.woa/ws/genres"
ITUNES_CHART_PATHS = {
    "top-free": "topfreeapplications",
    "top-paid": "toppaidapplications",
    "top-grossing": "topgrossingapplications",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AppleStoreScraper/2.0; +https://discovering-apps-jack.streamlit.app)"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Apple App Store metadata (snapshot preserving)."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--search-term",
        help="Keyword to search for within the App Store.",
    )
    source_group.add_argument(
        "--collection",
        choices=["top-free", "top-paid", "top-grossing"],
        help="Fetch the specified top chart collection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum apps to fetch (1-200). Default: 100.",
    )
    parser.add_argument(
        "--country",
        default="us",
        help="Two-letter country code for the App Store. Default: us.",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="When set with --collection, iterate every category to capture rankings.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("exports") / "app_store_apps_v2.db",
        help="Destination SQLite database. Default: exports/app_store_apps_v2.db",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=100,
        help="Batch size when requesting detailed metadata.",
    )
    parser.add_argument(
        "--note",
        help="Optional note stored with this scrape run for later reference.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    args = parser.parse_args()

    if args.all_categories and not args.collection:
        parser.error("--all-categories requires --collection.")
    return args


def bounded_limit(limit: int) -> int:
    return max(1, min(limit, 200))


def fetch_search_results(term: str, country: str, limit: int) -> List[Dict]:
    params = {
        "term": term,
        "country": country,
        "entity": "software",
        "limit": bounded_limit(limit),
    }
    logging.debug("Requesting search results from %s with %s", SEARCH_API_URL, params)
    resp = requests.get(SEARCH_API_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results", [])
    logging.info("Fetched %d search results for '%s'", len(results), term)
    return results


def fetch_collection_ids(collection: str, country: str, limit: int) -> List[str]:
    url = RSS_COLLECTION_URL.format(
        country=country,
        collection=collection,
        limit=bounded_limit(limit),
    )
    logging.debug("Requesting collection feed %s", url)
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    resp.raise_for_status()
    feed = resp.json().get("feed", {})
    results = feed.get("results", [])
    ids = [entry["id"] for entry in results if "id" in entry]
    logging.info("Fetched %d app ids for collection '%s'", len(ids), collection)
    return ids


def fetch_category_map(country: str) -> Dict[str, str]:
    params = {"id": "36", "cc": country}
    logging.debug("Requesting genre tree with %s", params)
    resp = requests.get(GENRE_TREE_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    root_node = payload.get("36") or next(iter(payload.values()), {})
    subgenres = root_node.get("subgenres", {})
    return {genre_id: info.get("name", genre_id) for genre_id, info in subgenres.items()}


def fetch_category_chart_ids(
    collection: str, country: str, limit: int, genre_id: str
) -> List[int]:
    chart_path = ITUNES_CHART_PATHS.get(collection)
    if not chart_path:
        raise ValueError(f"Unsupported collection '{collection}' for category scrape.")
    url = (
        f"https://itunes.apple.com/{country}/rss/{chart_path}/genre={genre_id}/limit={bounded_limit(limit)}/json"
    )
    logging.debug("Requesting category chart %s", url)
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    resp.raise_for_status()
    feed = resp.json().get("feed", {})
    entries = feed.get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]
    ids: List[int] = []
    for entry in entries:
        entry_id = entry.get("id", {}).get("attributes", {}).get("im:id")
        if not entry_id:
            continue
        try:
            ids.append(int(entry_id))
        except ValueError:
            logging.debug("Skipping non-numeric app id '%s'", entry_id)
    return ids


def build_category_memberships(
    collection: str,
    country: str,
    limit: int,
) -> Tuple[Dict[int, List[Dict[str, object]]], List[Dict[str, object]]]:
    categories = fetch_category_map(country)
    membership_map: Dict[int, List[Dict[str, object]]] = {}
    ranking_rows: List[Dict[str, object]] = []
    for genre_id, name in sorted(categories.items(), key=lambda item: str(item[1]).lower()):
        ids = fetch_category_chart_ids(collection, country, limit, genre_id)
        if not ids:
            logging.debug("No ids for category '%s' (%s)", name, genre_id)
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
    return membership_map, ranking_rows


def chunked(iterable: Iterable[object], size: int) -> Iterable[List[object]]:
    chunk: List[object] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def lookup_apps(app_ids: Sequence[object], country: str, batch_size: int) -> List[Dict]:
    all_results: List[Dict] = []
    for batch in chunked(app_ids, batch_size):
        params = {"id": ",".join(str(app_id) for app_id in batch), "country": country, "entity": "software"}
        logging.debug("Lookup request for %s", params["id"])
        resp = requests.get(LOOKUP_API_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        all_results.extend(payload.get("results", []))
    return all_results


def ensure_db_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            country TEXT,
            collection TEXT,
            search_term TEXT,
            limit_requested INTEGER,
            all_categories INTEGER,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS app_snapshots (
            run_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            name TEXT,
            description TEXT,
            release_date TEXT,
            current_version_release_date TEXT,
            version TEXT,
            primary_genre_id INTEGER,
            primary_genre_name TEXT,
            genre_ids TEXT,
            genres TEXT,
            content_advisory_rating TEXT,
            price REAL,
            formatted_price TEXT,
            currency TEXT,
            is_free INTEGER,
            has_in_app_purchases INTEGER,
            seller_name TEXT,
            seller_url TEXT,
            developer_id TEXT,
            bundle_id TEXT,
            average_user_rating REAL,
            average_user_rating_current REAL,
            user_rating_count INTEGER,
            user_rating_count_current INTEGER,
            rating_count_list TEXT,
            language_codes TEXT,
            minimum_os_version TEXT,
            file_size_bytes INTEGER,
            screenshot_urls TEXT,
            ipad_screenshot_urls TEXT,
            appletv_screenshot_urls TEXT,
            app_store_url TEXT,
            artwork_url TEXT,
            chart_memberships TEXT,
            scraped_at TEXT,
            PRIMARY KEY (run_id, track_id),
            FOREIGN KEY (run_id) REFERENCES scrape_runs(id)
        );

        CREATE TABLE IF NOT EXISTS app_rankings (
            run_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            chart_type TEXT NOT NULL,
            category_id TEXT,
            category_name TEXT,
            rank INTEGER,
            PRIMARY KEY (run_id, track_id, chart_type, category_id),
            FOREIGN KEY (run_id) REFERENCES scrape_runs(id)
        );
        """
    )


def insert_run(
    connection: sqlite3.Connection,
    *,
    source: str,
    country: str,
    collection: Optional[str],
    search_term: Optional[str],
    limit_requested: int,
    all_categories: bool,
    note: Optional[str],
) -> int:
    ensure_db_schema(connection)
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with connection:
        cursor = connection.execute(
            """
            INSERT INTO scrape_runs (
                created_at, source, country, collection,
                search_term, limit_requested, all_categories, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                source,
                country,
                collection,
                search_term,
                limit_requested,
                int(all_categories),
                note,
            ),
        )
    return int(cursor.lastrowid)


def normalize_snapshot(
    record: Dict,
    run_id: int,
    memberships: Optional[List[Dict[str, object]]],
    scraped_at: str,
) -> Dict[str, object]:
    screenshot_urls = record.get("screenshotUrls") or []
    ipad_screens = record.get("ipadScreenshotUrls") or []
    appletv_screens = record.get("appletvScreenshotUrls") or []
    genres = record.get("genres") or []
    genre_ids = record.get("genreIds") or []
    language_codes = record.get("languageCodesISO2A") or record.get("languageCodesISO2", [])
    rating_counts = record.get("userRatingCountForCurrentVersion") or record.get("ratingCountList")
    if isinstance(rating_counts, list):
        rating_count_json = json.dumps(rating_counts)
    else:
        rating_count_json = None

    return {
        "run_id": run_id,
        "track_id": record.get("trackId"),
        "name": record.get("trackName"),
        "description": record.get("description"),
        "release_date": record.get("releaseDate"),
        "current_version_release_date": record.get("currentVersionReleaseDate"),
        "version": record.get("version"),
        "primary_genre_id": record.get("primaryGenreId"),
        "primary_genre_name": record.get("primaryGenreName"),
        "genre_ids": json.dumps(genre_ids),
        "genres": json.dumps(genres),
        "content_advisory_rating": record.get("contentAdvisoryRating"),
        "price": record.get("price"),
        "formatted_price": record.get("formattedPrice"),
        "currency": record.get("currency"),
        "is_free": int((record.get("price") or 0) == 0),
        "has_in_app_purchases": int(bool(record.get("features")) or bool(record.get("inAppPurchases"))),
        "seller_name": record.get("sellerName"),
        "seller_url": record.get("sellerUrl"),
        "developer_id": record.get("artistId"),
        "bundle_id": record.get("bundleId"),
        "average_user_rating": record.get("averageUserRating"),
        "average_user_rating_current": record.get("averageUserRatingForCurrentVersion"),
        "user_rating_count": record.get("userRatingCount"),
        "user_rating_count_current": record.get("userRatingCountForCurrentVersion"),
        "rating_count_list": rating_count_json,
        "language_codes": json.dumps(language_codes),
        "minimum_os_version": record.get("minimumOsVersion"),
        "file_size_bytes": record.get("fileSizeBytes"),
        "screenshot_urls": json.dumps(screenshot_urls),
        "ipad_screenshot_urls": json.dumps(ipad_screens),
        "appletv_screenshot_urls": json.dumps(appletv_screens),
        "app_store_url": record.get("trackViewUrl"),
        "artwork_url": record.get("artworkUrl100") or record.get("artworkUrl512"),
        "chart_memberships": json.dumps(memberships or []),
        "scraped_at": scraped_at,
    }


def insert_snapshots(connection: sqlite3.Connection, snapshots: Sequence[Dict[str, object]]) -> None:
    if not snapshots:
        return
    with connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO app_snapshots (
                run_id, track_id, name, description,
                release_date, current_version_release_date, version,
                primary_genre_id, primary_genre_name, genre_ids, genres,
                content_advisory_rating, price, formatted_price, currency,
                is_free, has_in_app_purchases, seller_name, seller_url,
                developer_id, bundle_id, average_user_rating,
                average_user_rating_current, user_rating_count,
                user_rating_count_current, rating_count_list, language_codes,
                minimum_os_version, file_size_bytes, screenshot_urls,
                ipad_screenshot_urls, appletv_screenshot_urls, app_store_url,
                artwork_url, chart_memberships, scraped_at
            ) VALUES (
                :run_id, :track_id, :name, :description,
                :release_date, :current_version_release_date, :version,
                :primary_genre_id, :primary_genre_name, :genre_ids, :genres,
                :content_advisory_rating, :price, :formatted_price, :currency,
                :is_free, :has_in_app_purchases, :seller_name, :seller_url,
                :developer_id, :bundle_id, :average_user_rating,
                :average_user_rating_current, :user_rating_count,
                :user_rating_count_current, :rating_count_list, :language_codes,
                :minimum_os_version, :file_size_bytes, :screenshot_urls,
                :ipad_screenshot_urls, :appletv_screenshot_urls, :app_store_url,
                :artwork_url, :chart_memberships, :scraped_at
            )
            """,
            snapshots,
        )


def insert_rankings(
    connection: sqlite3.Connection,
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
    with connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO app_rankings (
                run_id, track_id, chart_type, category_id, category_name, rank
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def collect_raw_records(
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict[int, List[Dict[str, object]]], List[Dict[str, object]], str]:
    if args.search_term:
        raw_records = fetch_search_results(args.search_term, args.country, args.limit)
        membership_map: Dict[int, List[Dict[str, object]]] = {}
        ranking_rows: List[Dict[str, object]] = []
        source = "search"
    elif args.all_categories:
        membership_map, ranking_rows = build_category_memberships(
            args.collection,
            args.country,
            args.limit,
        )
        raw_records = lookup_apps(list(membership_map.keys()), args.country, args.lookup_batch_size)
        source = "collection-all-categories"
    else:
        ids = fetch_collection_ids(args.collection, args.country, args.limit)
        membership_map = {
            int(app_id): [
                {
                    "chart_type": args.collection,
                    "category_id": None,
                    "category_name": args.collection,
                    "rank": rank,
                }
            ]
            for rank, app_id in enumerate(ids, start=1)
        }
        ranking_rows = [
            {
                "track_id": int(app_id),
                "chart_type": args.collection,
                "category_id": None,
                "category_name": args.collection,
                "rank": rank,
            }
            for rank, app_id in enumerate(ids, start=1)
        ]
        raw_records = lookup_apps(ids, args.country, args.lookup_batch_size)
        source = "collection"
    logging.info("Collected %d detailed app records.", len(raw_records))
    return raw_records, membership_map, ranking_rows, source


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    raw_records, membership_map, ranking_rows, source = collect_raw_records(args)
    if not raw_records:
        logging.warning("No app metadata retrieved.")
        return

    scraped_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output_db)

    run_id = insert_run(
        connection,
        source=source,
        country=args.country,
        collection=args.collection,
        search_term=args.search_term,
        limit_requested=args.limit,
        all_categories=args.all_categories,
        note=args.note,
    )
    logging.info("Created scrape run %s", run_id)

    snapshots: List[Dict[str, object]] = []
    for record in raw_records:
        track_id = record.get("trackId")
        if not track_id:
            continue
        memberships = membership_map.get(int(track_id))
        snapshot = normalize_snapshot(record, run_id, memberships, scraped_at)
        snapshots.append(snapshot)

    insert_snapshots(connection, snapshots)
    insert_rankings(connection, run_id, ranking_rows)
    connection.close()
    logging.info(
        "Persisted %d app snapshots (run %s) to %s",
        len(snapshots),
        run_id,
        args.output_db.resolve(),
    )


if __name__ == "__main__":
    main()

