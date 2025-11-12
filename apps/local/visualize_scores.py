#!/usr/bin/env python3
"""Render scatter plots for Stage 2 success scores vs build time estimates."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DB_PATH = Path("exports") / "app_store_apps.db"
DEFAULT_OUTPUT_PATH = Path("visualizations") / "success_vs_build_time.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success score against build time estimates for scraped apps."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination image file for the scatter plot (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=0,
        help="Filter out apps with fewer than this many ratings (default: 0).",
    )
    parser.add_argument(
        "--max-build-time",
        type=float,
        help="Optionally filter out apps with build_time_estimate greater than this value.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively instead of writing to disk.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def load_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT
                track_id,
                name,
                category,
                build_time_estimate,
                success_score,
                number_of_ratings,
                review_score,
                price,
                developer
            FROM apps
            WHERE build_time_estimate IS NOT NULL
              AND success_score IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
    if df.empty:
        raise ValueError("No rows with both build_time_estimate and success_score found.")
    return df


def preprocess(df: pd.DataFrame, *, min_ratings: int, max_build_time: float | None) -> pd.DataFrame:
    filtered = df.copy()
    if min_ratings > 0:
        filtered = filtered[filtered["number_of_ratings"].fillna(0) >= min_ratings]
    if max_build_time is not None:
        filtered = filtered[filtered["build_time_estimate"] <= max_build_time]
    filtered = filtered[filtered["build_time_estimate"] > 0]
    if filtered.empty:
        raise ValueError("No rows remain after applying filters.")
    filtered["success_per_week"] = (
        filtered["success_score"] / filtered["build_time_estimate"]
    )
    filtered.sort_values("success_per_week", ascending=False, inplace=True)
    return filtered


def render_scatter(df: pd.DataFrame, *, output: Path, show: bool) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    categories = df["category"].fillna("Unknown")
    unique_categories = categories.unique()
    cmap = plt.get_cmap("tab20")

    for idx, category in enumerate(unique_categories):
        subset = df[categories == category]
        ax.scatter(
            subset["build_time_estimate"],
            subset["success_score"],
            s=40 + subset["number_of_ratings"].fillna(0).pow(0.5) * 2,
            alpha=0.7,
            label=category,
            color=cmap(idx % cmap.N),
        )

    ax.set_xlabel("Estimated Build Time (weeks)")
    ax.set_ylabel("Success Score (0-100)")
    ax.set_title("Success vs Build Effort for Apple App Store Apps")

    # Annotate top performers by success-per-week ratio
    top_performers = df.head(10)
    for _, row in top_performers.iterrows():
        ax.annotate(
            row["name"][:30],
            (row["build_time_estimate"], row["success_score"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.legend(
        title="Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    if show:
        plt.show()
    else:
        fig.savefig(output, dpi=150)
        logging.info("Saved scatter plot to %s", output.resolve())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df = load_dataframe(args.db_path)
    logging.info("Loaded %d apps with Stage 2 scores.", len(df))

    df_filtered = preprocess(
        df,
        min_ratings=args.min_ratings,
        max_build_time=args.max_build_time,
    )
    logging.info("Filtered down to %d apps after applying constraints.", len(df_filtered))

    render_scatter(
        df_filtered,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
