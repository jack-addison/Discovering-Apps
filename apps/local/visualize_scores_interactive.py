#!/usr/bin/env python3
"""Interactive visualization of Stage 2 success vs build time scores."""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.express as px

DEFAULT_DB_PATH = Path("exports") / "app_store_apps.db"
DEFAULT_OUTPUT = Path("visualizations") / "success_vs_build_time.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive scatter plot for success score vs build time."
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
        default=DEFAULT_OUTPUT,
        help=f"Destination HTML file for the interactive chart (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=0,
        help="Filter out apps with fewer than this many user ratings (default: 0).",
    )
    parser.add_argument(
        "--max-build-time",
        type=float,
        help="Filter out apps with build_time_estimate greater than this threshold.",
    )
    parser.add_argument(
        "--min-success",
        type=float,
        help="Filter out apps with success_score lower than this threshold.",
    )
    parser.add_argument(
        "--quick-wins-only",
        action="store_true",
        help="Display only apps within the quick wins zone (build ≤12 weeks and success ≥70).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the generated HTML in your default browser.",
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
                number_of_downloads,
                price,
                developer,
                chart_memberships
            FROM apps
            WHERE build_time_estimate IS NOT NULL
              AND success_score IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
    if df.empty:
        raise ValueError("No rows with Stage 2 scores found in the database.")
    return df


def preprocess(
    df: pd.DataFrame,
    *,
    min_ratings: int,
    max_build_time: float | None,
    min_success: float | None,
) -> tuple[pd.DataFrame, int]:
    working = df.copy()
    working = working[working["build_time_estimate"] > 0]
    if min_ratings > 0:
        working = working[working["number_of_ratings"].fillna(0) >= min_ratings]
    if max_build_time is not None:
        working = working[working["build_time_estimate"] <= max_build_time]
    if min_success is not None:
        working = working[working["success_score"] >= min_success]
    if working.empty:
        raise ValueError("No rows remain after applying filters.")

    working["success_per_week"] = (
        working["success_score"] / working["build_time_estimate"]
    )
    working["confidence_proxy"] = working["number_of_ratings"].fillna(0).apply(
        lambda x: math.log10(x + 1)
    )
    working["bubble_size"] = working["confidence_proxy"].apply(lambda x: 10 + x * 15)
    working["category_clean"] = working["category"].fillna("Unknown")
    working["price_tier"] = working["price"].fillna(0).apply(
        lambda value: "Paid" if value and value > 0 else "Free"
    )
    working["category_segment"] = (
        working["category_clean"] + " (" + working["price_tier"] + ")"
    )
    working["quick_win"] = (working["build_time_estimate"] <= 12) & (
        working["success_score"] >= 70
    )
    quick_counts = (
        working.groupby("category_segment")["quick_win"].sum().astype(int).to_dict()
    )
    working["category_label"] = working["category_segment"].apply(
        lambda name: f"{name} - ({quick_counts.get(name, 0)})"
    )
    working.sort_values("success_per_week", ascending=False, inplace=True)
    quick_win_count = int(working["quick_win"].sum())
    return working, quick_win_count


def build_figure(
    df: pd.DataFrame,
    quick_win_count: int,
) -> px.scatter:
    categories = df["category_label"].dropna().unique()
    palette = px.colors.qualitative.Plotly
    color_map = {
        category: palette[idx % len(palette)]
        for idx, category in enumerate(categories)
    }

    fig = px.scatter(
        df,
        x="build_time_estimate",
        y="success_score",
        color="category_label",
        size="bubble_size",
        hover_name="name",
        hover_data={
            "developer": True,
            "category_clean": True,
            "price_tier": True,
            "success_score": ":.1f",
            "build_time_estimate": ":.1f",
            "success_per_week": ":.2f",
            "number_of_ratings": True,
            "review_score": ":.2f",
            "price": True,
        },
        color_discrete_map=color_map,
    )
    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="Success Score vs Build Time (interactive)",
        xaxis_title="Estimated Build Time (weeks)",
        yaxis_title="Success Score (0-100)",
        legend_title="Category w/ price tier - (# quick wins)",
        template="plotly_white",
        margin=dict(l=60, r=60, t=70, b=60),
    )

    # Highlight the "high reward / low effort" quadrant (<=12 weeks, >=70 score) for reference.
    fig.add_shape(
        type="rect",
        x0=0,
        y0=70,
        x1=12,
        y1=100,
        fillcolor="rgba(0, 200, 150, 0.08)",
        line=dict(color="rgba(0, 200, 150, 0.3)", dash="dash"),
    )
    fig.add_annotation(
        x=6,
        y=95,
        text="Quick wins zone",
        showarrow=False,
        font=dict(color="rgba(0,120,90,0.9)", size=12),
    )
    return fig


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df = load_dataframe(args.db_path)
    logging.info("Loaded %d scored apps from %s", len(df), args.db_path)

    df = preprocess(
        df,
        min_ratings=args.min_ratings,
        max_build_time=args.max_build_time,
        min_success=args.min_success,
    )
    df, quick_win_count = df
    if args.quick_wins_only:
        df = df[df["quick_win"]].copy()
        logging.info("Applied quick wins filter; %d apps remain.", len(df))

    logging.info(
        "Remaining apps after filters: %d total (%d quick wins).",
        len(df),
        quick_win_count,
    )

    figure = build_figure(df, quick_win_count)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    logging.info("Interactive chart saved to %s", output_path.resolve())

    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
