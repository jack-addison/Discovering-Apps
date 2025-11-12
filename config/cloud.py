"""Shared configuration and helpers for SQLiteCloud connections."""

from __future__ import annotations

import time
from typing import Optional

import sqlitecloud

# Add a generous timeout to reduce TLS handshake failures on repeated runs.
CONNECTION_URI = (
    "sqlitecloud://cky1wteehz.g4.sqlite.cloud:8860/app_store_apps_v2.db"
    "?timeout=60&apikey=HuDMLrUJedgC54VQOfbfQ489AY0aDivJ6XXfe1r01Wo"
)


def connect(
    *,
    uri: Optional[str] = None,
    retries: int = 3,
    backoff_seconds: int = 5,
) -> sqlitecloud.dbapi2.Connection:
    """Return a SQLiteCloud connection with simple retry/back-off."""

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return sqlitecloud.connect(uri or CONNECTION_URI)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
    assert last_err is not None  # for type checkers
    raise last_err
