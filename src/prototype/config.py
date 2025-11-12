"""Configuration helpers for the prototype PostgreSQL workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class MissingDatabaseUrl(RuntimeError):
    """Raised when the required PostgreSQL connection string is absent."""


@dataclass(frozen=True)
class PrototypeSettings:
    """Simple container for prototype environment configuration."""

    postgres_dsn: str
    sqlite_path: str


def load_settings(
    *,
    postgres_env_var: str = "PROTOTYPE_DATABASE_URL",
    sqlite_default: str = "exports/app_store_apps_v2.db",
) -> PrototypeSettings:
    """Load configuration for the prototype PostgreSQL tooling.

    Parameters
    ----------
    postgres_env_var:
        Environment variable that contains the Neon / PostgreSQL connection string.
    sqlite_default:
        Fallback path to the source SQLite database to migrate from.
    """

    postgres_dsn: Optional[str] = os.getenv(postgres_env_var)
    if not postgres_dsn:
        raise MissingDatabaseUrl(
            f"Environment variable {postgres_env_var} must be set with the Neon DSN."
        )
    return PrototypeSettings(
        postgres_dsn=postgres_dsn,
        sqlite_path=os.getenv("PROTOTYPE_SQLITE_PATH", sqlite_default),
    )

