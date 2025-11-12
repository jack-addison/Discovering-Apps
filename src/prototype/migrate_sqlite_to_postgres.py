#!/usr/bin/env python3
"""Copy the existing SQLite dataset into a Neon/PostgreSQL instance."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

from .config import PrototypeSettings, load_settings

SQLiteRow = sqlite3.Row
RowData = Tuple[object, ...]


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Convert ISO-ish strings from SQLite into timezone-aware datetimes."""

    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_bool(value: Optional[object]) -> Optional[bool]:
    """Convert SQLite truthy values (0/1, strings) into booleans."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "null"}:
            return None
        if text in {"true", "t", "yes", "y"}:
            return True
        if text in {"false", "f", "no", "n"}:
            return False
        try:
            return bool(int(text))
        except ValueError:
            return None
    return None


def parse_float(value: Optional[object]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TableSpec:
    name: str
    columns: Sequence[str]
    select_sql: str
    transform: Callable[[SQLiteRow], RowData]
    serial_column: Optional[str] = None  # to reset sequences after insertion


def iter_rows(cursor: sqlite3.Cursor, batch_size: int) -> Iterable[List[SQLiteRow]]:
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def load_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    spec: TableSpec,
    *,
    batch_size: int,
) -> int:
    sqlite_cursor = sqlite_conn.execute(spec.select_sql)
    sqlite_cursor.row_factory = sqlite3.Row

    column_list = ", ".join(spec.columns)
    placeholders = ", ".join(["%s"] * len(spec.columns))
    insert_sql = (
        f"INSERT INTO {spec.name} ({column_list}) VALUES ({placeholders}) "
        "ON CONFLICT DO NOTHING"
    )

    inserted = 0
    with pg_conn.cursor() as pg_cur:
        for rows in iter_rows(sqlite_cursor, batch_size):
            payload = [spec.transform(row) for row in rows]
            if not payload:
                continue
            pg_cur.executemany(insert_sql, payload)
            inserted += len(payload)
    return inserted


def reset_serial(conn: psycopg.Connection, table: str, column: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT pg_get_serial_sequence(%s, %s);"),
            (table, column),
        )
        sequence_row = cur.fetchone()
        if not sequence_row or sequence_row[0] is None:
            return
        sequence = sequence_row[0]
        cur.execute(
            sql.SQL("SELECT COALESCE(MAX({column}), 0) FROM {table};").format(
                column=sql.Identifier(column),
                table=sql.Identifier(table),
            )
        )
        max_value = cur.fetchone()[0] or 0
        cur.execute("SELECT setval(%s, %s, true);", (sequence, max_value))


def ensure_schema(pg_conn: psycopg.Connection, schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle, pg_conn.cursor() as cur:
        cur.execute(handle.read())


def truncate_all(pg_conn: psycopg.Connection) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            TRUNCATE TABLE
                app_snapshot_cluster_members,
                app_snapshot_clusters,
                app_snapshot_neighbors,
                app_snapshot_embeddings,
                app_rankings,
                app_snapshot_deltas,
                app_snapshots,
                scrape_runs
            RESTART IDENTITY CASCADE;
            """
        )


def build_specs() -> Sequence[TableSpec]:
    return [
        TableSpec(
            name="scrape_runs",
            columns=(
                "id",
                "created_at",
                "source",
                "country",
                "collection",
                "search_term",
                "limit_requested",
                "all_categories",
                "note",
            ),
            select_sql="SELECT * FROM scrape_runs ORDER BY id",
            transform=lambda row: (
                row["id"],
                parse_timestamp(row["created_at"]),
                row["source"],
                row["country"],
                row["collection"],
                row["search_term"],
                row["limit_requested"],
                parse_bool(row["all_categories"]),
                row["note"],
            ),
            serial_column="id",
        ),
        TableSpec(
            name="app_snapshots",
            columns=(
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
                "build_time_estimate",
                "success_score",
                "success_reasoning",
            ),
            select_sql="SELECT * FROM app_snapshots ORDER BY run_id, track_id",
            transform=lambda row: (
                row["run_id"],
                row["track_id"],
                row["name"],
                row["description"],
                parse_timestamp(row["release_date"]),
                parse_timestamp(row["current_version_release_date"]),
                row["version"],
                row["primary_genre_id"],
                row["primary_genre_name"],
                row["genre_ids"],
                row["genres"],
                row["content_advisory_rating"],
                parse_float(row["price"]),
                row["formatted_price"],
                row["currency"],
                parse_bool(row["is_free"]),
                parse_bool(row["has_in_app_purchases"]),
                row["seller_name"],
                row["seller_url"],
                row["developer_id"],
                row["bundle_id"],
                parse_float(row["average_user_rating"]),
                parse_float(row["average_user_rating_current"]),
                row["user_rating_count"],
                row["user_rating_count_current"],
                row["rating_count_list"],
                row["language_codes"],
                row["minimum_os_version"],
                row["file_size_bytes"],
                row["screenshot_urls"],
                row["ipad_screenshot_urls"],
                row["appletv_screenshot_urls"],
                row["app_store_url"],
                row["artwork_url"],
                row["chart_memberships"],
                parse_timestamp(row["scraped_at"]),
                parse_float(row["build_time_estimate"]),
                parse_float(row["success_score"]),
                row["success_reasoning"],
            ),
        ),
        TableSpec(
            name="app_rankings",
            columns=(
                "run_id",
                "track_id",
                "chart_type",
                "category_id",
                "category_name",
                "rank",
            ),
            select_sql="SELECT * FROM app_rankings ORDER BY run_id, track_id",
            transform=lambda row: (
                row["run_id"],
                row["track_id"],
                row["chart_type"],
                row["category_id"],
                row["category_name"],
                row["rank"],
            ),
        ),
        TableSpec(
            name="app_snapshot_embeddings",
            columns=(
                "run_id",
                "track_id",
                "model",
                "description_sha256",
                "embedding_json",
                "vector_length",
                "created_at",
            ),
            select_sql="SELECT * FROM app_snapshot_embeddings ORDER BY run_id, track_id",
            transform=lambda row: (
                row["run_id"],
                row["track_id"],
                row["model"],
                row["description_sha256"],
                row["embedding_json"],
                row["vector_length"],
                parse_timestamp(row["created_at"]),
            ),
        ),
        TableSpec(
            name="app_snapshot_neighbors",
            columns=(
                "run_id",
                "track_id",
                "neighbor_run_id",
                "neighbor_track_id",
                "model",
                "similarity",
                "rank",
                "created_at",
            ),
            select_sql="SELECT * FROM app_snapshot_neighbors ORDER BY run_id, track_id",
            transform=lambda row: (
                row["run_id"],
                row["track_id"],
                row["neighbor_run_id"],
                row["neighbor_track_id"],
                row["model"],
                parse_float(row["similarity"]),
                row["rank"],
                parse_timestamp(row["created_at"]),
            ),
        ),
        TableSpec(
            name="app_snapshot_clusters",
            columns=(
                "id",
                "scope",
                "model",
                "label",
                "keywords_json",
                "size",
                "avg_success",
                "avg_build",
                "avg_demand",
                "created_at",
            ),
            select_sql="SELECT * FROM app_snapshot_clusters ORDER BY id",
            transform=lambda row: (
                row["id"],
                row["scope"],
                row["model"],
                row["label"],
                row["keywords_json"],
                row["size"],
                parse_float(row["avg_success"]),
                parse_float(row["avg_build"]),
                parse_float(row["avg_demand"]),
                parse_timestamp(row["created_at"]),
            ),
            serial_column="id",
        ),
        TableSpec(
            name="app_snapshot_cluster_members",
            columns=(
                "cluster_id",
                "run_id",
                "track_id",
                "distance",
            ),
            select_sql="SELECT * FROM app_snapshot_cluster_members ORDER BY cluster_id, run_id, track_id",
            transform=lambda row: (
                row["cluster_id"],
                row["run_id"],
                row["track_id"],
                parse_float(row["distance"]),
            ),
        ),
        TableSpec(
            name="app_snapshot_deltas",
            columns=(
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
            ),
            select_sql="SELECT * FROM app_snapshot_deltas ORDER BY run_id, track_id",
            transform=lambda row: (
                row["run_id"],
                parse_timestamp(row["run_created_at"]),
                row["track_id"],
                row["name"],
                row["category"],
                parse_float(row["price"]),
                row["currency"],
                parse_float(row["average_user_rating"]),
                row["user_rating_count"],
                parse_float(row["build_time_estimate"]),
                parse_float(row["success_score"]),
                row["success_reasoning"],
                parse_float(row["best_rank"]),
                row["description"],
                row["version"],
                row["developer"],
                row["prev_run_id"],
                parse_timestamp(row["prev_run_created_at"]),
                parse_float(row["prev_success_score"]),
                parse_float(row["prev_build_time"]),
                parse_float(row["prev_rating"]),
                parse_float(row["prev_rating_count"]),
                parse_float(row["prev_price"]),
                row["prev_currency"],
                parse_float(row["prev_rank"]),
                parse_bool(row["is_new_track"]),
                parse_float(row["delta_success"]),
                parse_float(row["delta_build_time"]),
                parse_float(row["delta_rating"]),
                parse_float(row["delta_rating_count"]),
                parse_float(row["delta_price"]),
                parse_float(row["delta_rank"]),
                parse_bool(row["price_changed"]),
                parse_float(row["days_since_prev"]),
            ),
        ),
    ]


def migrate(settings: PrototypeSettings, *, batch_size: int, ensure_tables: bool, truncate: bool) -> None:
    sqlite_path = Path(settings.sqlite_path).expanduser().resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")

    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row

    with psycopg_connection(settings.postgres_dsn) as pg_conn:
        if ensure_tables:
            ensure_schema(pg_conn, Path("exports/schema_postgres.sql").resolve())
        if truncate:
            truncate_all(pg_conn)

        total_inserted = {}
        specs = build_specs()
        for spec in specs:
            count = load_table(sqlite_conn, pg_conn, spec, batch_size=batch_size)
            total_inserted[spec.name] = count
            if spec.serial_column:
                reset_serial(pg_conn, spec.name, spec.serial_column)
        pg_conn.commit()

    sqlite_conn.close()

    for table, count in total_inserted.items():
        print(f"{table}: inserted {count:,} rows")


def psycopg_connection(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, row_factory=tuple_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate the local SQLite dataset into the Neon/PostgreSQL prototype environment."
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("exports") / "app_store_apps_v2.db",
        help="Path to the source SQLite database (default: exports/app_store_apps_v2.db).",
    )
    parser.add_argument(
        "--postgres-dsn",
        type=str,
        help="Neon/PostgreSQL connection string. Overrides PROTOTYPE_DATABASE_URL when provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to insert per batch when copying tables (default: 1000).",
    )
    parser.add_argument(
        "--no-ensure-schema",
        action="store_true",
        help="Skip applying exports/schema_postgres.sql before inserting data.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Skip truncating destination tables before inserting rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_settings = load_settings()
    settings = PrototypeSettings(
        postgres_dsn=args.postgres_dsn or base_settings.postgres_dsn,
        sqlite_path=str(args.sqlite_path),
    )

    migrate(
        settings,
        batch_size=args.batch_size,
        ensure_tables=not args.no_ensure_schema,
        truncate=not args.no_truncate,
    )


if __name__ == "__main__":
    main()
