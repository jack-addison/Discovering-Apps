#!/usr/bin/env python3
"""Harvest metadata for Apple App Store listings into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
        "Mozilla/5.0 (compatible; AppleStoreScraper/1.0; +https://github.com/openai/codex)"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Apple App Store metadata and persist it to SQLite."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--search-term",
        help="Keyword to search for within the App Store (uses iTunes Search API).",
    )
    source_group.add_argument(
        "--collection",
        choices=["top-free", "top-paid", "top-grossing"],
        help="Fetch the specified top chart collection instead of a keyword search.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of apps to fetch (1-200). Default: 100.",
    )
    parser.add_argument(
        "--country",
        default="us",
        help="Two-letter App Store country code. Default: us.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("exports") / "app_store_apps.db",
        help="Destination SQLite database path. Default: exports/app_store_apps.db.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=100,
        help="Batch size when requesting detailed app metadata via lookup API.",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Iterate through every App Store category and gather the selected chart.",
    )
    args = parser.parse_args()

    if args.all_categories and args.search_term:
        parser.error("--all-categories cannot be combined with --search-term.")

    if not args.search_term and not args.collection:
        if args.all_categories:
            args.collection = "top-free"
            logging.debug("Defaulting chart collection to 'top-free' for all categories run.")
        else:
            parser.error("Provide --search-term or --collection to select a data source.")

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
    response = requests.get(
        SEARCH_API_URL, params=params, headers=DEFAULT_HEADERS, timeout=20
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    logging.info("Fetched %d search results for term '%s'", len(results), term)
    return results


def fetch_collection_ids(collection: str, country: str, limit: int) -> List[str]:
    url = RSS_COLLECTION_URL.format(
        country=country, collection=collection, limit=bounded_limit(limit)
    )
    logging.debug("Requesting collection feed from %s", url)
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    response.raise_for_status()
    feed = response.json().get("feed", {})
    results = feed.get("results", [])
    ids = [entry["id"] for entry in results if "id" in entry]
    logging.info("Fetched %d app ids from %s", len(ids), collection)
    return ids


def fetch_category_map(country: str) -> Dict[str, str]:
    params = {"id": "36", "cc": country}
    logging.debug("Requesting App Store genre tree from %s with %s", GENRE_TREE_URL, params)
    response = requests.get(GENRE_TREE_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    root_node = payload.get("36") or next(iter(payload.values()), {})
    subgenres = root_node.get("subgenres", {})
    categories = {genre_id: info.get("name", genre_id) for genre_id, info in subgenres.items()}
    logging.info("Discovered %d App Store categories for country '%s'", len(categories), country)
    return categories


def fetch_category_chart_ids(
    collection: str, country: str, limit: int, genre_id: str
) -> List[int]:
    chart_path = ITUNES_CHART_PATHS.get(collection)
    if not chart_path:
        raise ValueError(f"Unsupported collection '{collection}' for category scraping.")
    url = (
        f"https://itunes.apple.com/{country}/rss/{chart_path}/genre={genre_id}/limit={bounded_limit(limit)}/json"
    )
    logging.debug("Requesting category chart from %s", url)
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    response.raise_for_status()
    feed = response.json().get("feed", {})
    entries = feed.get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]
    ids: List[int] = []
    for entry in entries:
        entry_id = (
            entry.get("id", {})
            .get("attributes", {})
            .get("im:id")
        )
        if not entry_id:
            continue
        try:
            ids.append(int(entry_id))
        except ValueError:
            logging.debug("Skipping non-numeric app id '%s' from feed", entry_id)
    logging.info(
        "Fetched %d app ids for genre %s while scraping '%s'",
        len(ids),
        genre_id,
        collection,
    )
    return ids


def build_category_memberships(
    collection: str, country: str, limit: int
) -> Dict[int, List[Dict[str, object]]]:
    categories = fetch_category_map(country)
    memberships: Dict[int, List[Dict[str, object]]] = {}
    for genre_id, name in sorted(
        categories.items(), key=lambda item: str(item[1]).lower()
    ):
        ids = fetch_category_chart_ids(collection, country, limit, genre_id)
        if not ids:
            logging.warning("No apps returned for category '%s' (%s)", name, genre_id)
            continue
        for rank, track_id in enumerate(ids, start=1):
            memberships.setdefault(track_id, []).append(
                {
                    "chart_type": collection,
                    "category_id": genre_id,
                    "category_name": name,
                    "rank": rank,
                }
            )
    return memberships


def chunked(iterable: Iterable[object], size: int) -> Iterable[List[object]]:
    chunk: List[object] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def lookup_apps(app_ids: Iterable[object], country: str, batch_size: int) -> List[Dict]:
    all_results: List[Dict] = []
    for batch in chunked(app_ids, batch_size):
        id_list = [str(app_id) for app_id in batch]
        params = {"id": ",".join(id_list), "country": country, "entity": "software"}
        logging.debug("Lookup request for %s", params["id"])
        response = requests.get(
            LOOKUP_API_URL, params=params, headers=DEFAULT_HEADERS, timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        logging.info("Fetched %d detailed records via lookup", len(results))
        all_results.extend(results)
    return all_results


def normalize_app(
    record: Dict,
    memberships: Optional[List[Dict[str, object]]] = None,
) -> Optional[Dict]:
    track_id = record.get("trackId")
    name = record.get("trackName")
    if not track_id or not name:
        logging.debug("Skipping record missing trackId or trackName: %s", record)
        return None
    category = record.get("primaryGenreName")
    if not category:
        genres = record.get("genres") or []
        category = genres[0] if genres else None
    review_score = record.get("averageUserRating")
    description = record.get("description", "")
    rating_count = record.get("userRatingCount")
    # Apple does not disclose download counts; using rating count as a conservative proxy.
    downloads_proxy = rating_count
    serialized_memberships = json.dumps(memberships or [])
    normalized = {
        "track_id": track_id,
        "name": name,
        "category": category,
        "review_score": review_score,
        "description": description,
        "number_of_downloads": downloads_proxy,
        "number_of_ratings": rating_count,
        "developer": record.get("sellerName"),
        "price": record.get("price"),
        "currency": record.get("currency"),
        "language_codes": ",".join(record.get("languageCodesISO2A", [])),
        "app_store_url": record.get("trackViewUrl"),
        "artwork_url": record.get("artworkUrl100"),
        "chart_memberships": serialized_memberships,
        "scraped_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return normalized


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS apps (
            track_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            review_score REAL,
            description TEXT,
            number_of_downloads INTEGER,
            number_of_ratings INTEGER,
            developer TEXT,
            price REAL,
            currency TEXT,
            language_codes TEXT,
            app_store_url TEXT,
            artwork_url TEXT,
            chart_memberships TEXT,
            scraped_at TEXT
        )
        """
    )
    ensure_app_columns(connection)


def ensure_app_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(apps)")
    }
    required_columns = {
        "chart_memberships": "TEXT",
    }
    for column, column_type in required_columns.items():
        if column not in existing_columns:
            logging.debug("Adding missing column '%s' to apps table", column)
            connection.execute(f"ALTER TABLE apps ADD COLUMN {column} {column_type}")


def upsert_apps(connection: sqlite3.Connection, apps: Iterable[Dict]) -> int:
    init_db(connection)
    inserted = 0
    statement = """
        INSERT INTO apps (
            track_id, name, category, review_score, description,
            number_of_downloads, number_of_ratings, developer, price,
            currency, language_codes, app_store_url, artwork_url, chart_memberships,
            scraped_at
        ) VALUES (
            :track_id, :name, :category, :review_score, :description,
            :number_of_downloads, :number_of_ratings, :developer, :price,
            :currency, :language_codes, :app_store_url, :artwork_url, :chart_memberships,
            :scraped_at
        )
        ON CONFLICT(track_id) DO UPDATE SET
            name = excluded.name,
            category = excluded.category,
            review_score = excluded.review_score,
            description = excluded.description,
            number_of_downloads = excluded.number_of_downloads,
            number_of_ratings = excluded.number_of_ratings,
            developer = excluded.developer,
            price = excluded.price,
            currency = excluded.currency,
            language_codes = excluded.language_codes,
            app_store_url = excluded.app_store_url,
            artwork_url = excluded.artwork_url,
            chart_memberships = excluded.chart_memberships,
            scraped_at = excluded.scraped_at
    """
    with connection:
        for record in apps:
            connection.execute(statement, record)
            inserted += 1
    return inserted


def ensure_parent_directory(path: Path) -> None:
    if not path.parent.exists():
        logging.debug("Creating parent directory for database: %s", path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        membership_map: Dict[int, List[Dict[str, object]]] = {}
        if args.all_categories:
            membership_map = build_category_memberships(args.collection, args.country, args.limit)
            if not membership_map:
                logging.warning(
                    "No app ids returned while scraping all categories for '%s'",
                    args.collection,
                )
                return
            raw_records = lookup_apps(
                membership_map.keys(), args.country, args.lookup_batch_size
            )
        elif args.collection:
            app_ids = fetch_collection_ids(args.collection, args.country, args.limit)
            if not app_ids:
                logging.warning("No app ids returned for collection '%s'", args.collection)
                return
            raw_records = lookup_apps(app_ids, args.country, args.lookup_batch_size)
        else:
            raw_records = fetch_search_results(args.search_term, args.country, args.limit)

        normalized_records = []
        for raw in raw_records:
            memberships = None
            track_id = raw.get("trackId")
            if membership_map and track_id is not None:
                memberships = membership_map.get(int(track_id))
            normalized = normalize_app(raw, memberships=memberships)
            if normalized:
                normalized_records.append(normalized)

        if not normalized_records:
            logging.warning("No valid app records retrieved from the App Store.")
            return

        ensure_parent_directory(args.db_path)
        with sqlite3.connect(args.db_path) as conn:
            total = upsert_apps(conn, normalized_records)
        logging.info(
            "Persisted %d app records to %s",
            total,
            args.db_path.resolve(),
        )
    except requests.HTTPError as err:
        logging.error("HTTP error from Apple endpoints: %s", err)
    except requests.RequestException as err:
        logging.error("Network error while contacting Apple endpoints: %s", err)
    except sqlite3.DatabaseError as err:
        logging.error("SQLite error while writing results: %s", err)


if __name__ == "__main__":
    main()
