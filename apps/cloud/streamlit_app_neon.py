#!/usr/bin/env python3
"""Streamlit dashboard backed by a Neon PostgreSQL database."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Ensure repository root is importable when Streamlit runs this file as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.cloud import streamlit_app_cloud as cloud_ui  # type: ignore

NEON_DATABASE_URL = os.environ.get("PROTOTYPE_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_TTL_SECONDS = 900

AXIS_OPTIONS = {
    "build_time_estimate": "Estimated build time (weeks)",
    "success_score": "Success score (0-100)",
    "success_per_week": "Success per week",
    "number_of_ratings": "Rating count",
    "review_score": "Average review",
    "price": "Price (USD)",
    "number_of_downloads": "Download proxy (ratings)",
}


@st.cache_resource
def get_engine() -> Engine:
    if not NEON_DATABASE_URL:
        raise RuntimeError(
            "NEON_DATABASE_URL environment variable is not set. "
            "Set it to your Neon connection string before launching the app."
        )
    dsn = NEON_DATABASE_URL
    if dsn.startswith("postgres://"):
        # Normalise legacy postgres:// URLs for SQLAlchemy 2.x
        dsn = dsn.replace("postgres://", "postgresql+psycopg://", 1)
    elif dsn.startswith("postgresql://") and "+" not in dsn:
        # default to psycopg (v3) driver when no dialect specified
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(dsn, pool_pre_ping=True)


def fetch_dataframe(query: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params or {})
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_data() -> pd.DataFrame:
    query = """
        SELECT
            s.run_id,
            COALESCE(sr.created_at, s.scraped_at) AS run_created_at,
            s.track_id,
            s.name,
            s.primary_genre_name AS category,
            s.build_time_estimate,
            s.success_score,
            s.success_reasoning,
            s.user_rating_count AS number_of_ratings,
            s.average_user_rating AS review_score,
            s.user_rating_count AS number_of_downloads,
            s.price,
            s.currency,
            s.seller_name AS developer,
            s.description,
            s.language_codes,
            s.chart_memberships,
            s.scraped_at,
            s.is_free,
            s.has_in_app_purchases
        FROM app_snapshots s
        LEFT JOIN scrape_runs sr ON sr.id = s.run_id
        WHERE s.build_time_estimate IS NOT NULL
          AND s.success_score IS NOT NULL
    """
    df = fetch_dataframe(query)
    if df.empty:
        raise ValueError("No rows with Stage 2 scores found in the database.")

    df["price"] = df["price"].fillna(0.0)
    df["run_created_at"] = pd.to_datetime(
        df["run_created_at"].fillna(df["scraped_at"]), errors="coerce"
    )
    df["run_date"] = df["run_created_at"].dt.date
    df["run_created_at_str"] = df["run_created_at"].dt.strftime("%Y-%m-%d %H:%M")
    df["price_tier"] = np.where(df["price"] > 0, "Paid", "Free")
    df["category_clean"] = df["category"].fillna("Unknown")
    df["language_codes"] = df["language_codes"].apply(
        lambda codes: ", ".join(json.loads(codes)) if isinstance(codes, str) else None
    )
    df["chart_memberships"] = df["chart_memberships"].apply(
        lambda memberships: json.loads(memberships) if isinstance(memberships, str) else []
    )
    df["category_segment"] = df["category_clean"] + " (" + df["price_tier"] + ")"
    df["success_per_week"] = df["success_score"] / df["build_time_estimate"].replace(0, np.nan)
    df["confidence_proxy"] = np.log10(df["number_of_ratings"].fillna(0) + 1)
    df["bubble_size"] = 20 + df["confidence_proxy"] * 25
    df["quick_win"] = (df["build_time_estimate"] <= 12) & (df["success_score"] >= 70)
    df["demand_dissatisfaction"] = df["number_of_ratings"].fillna(0) * (5.0 - df["review_score"].fillna(5.0))
    diss_values = df["demand_dissatisfaction"].replace([np.inf, -np.inf], np.nan)
    if diss_values.notna().any():
        df["demand_dissatisfaction_percentile"] = diss_values.rank(pct=True) * 100
    else:
        df["demand_dissatisfaction_percentile"] = 0.0
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_run_catalog() -> pd.DataFrame:
    query = """
        SELECT
            r.id AS run_id,
            COALESCE(r.created_at, MIN(s.scraped_at)) AS run_created_at,
            r.source,
            r.note
        FROM scrape_runs r
        LEFT JOIN app_snapshots s ON s.run_id = r.id
        GROUP BY r.id, r.created_at, r.source, r.note
    """
    df = fetch_dataframe(query)
    if df.empty:
        return df
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    df["run_date"] = df["run_created_at"].dt.date
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_snapshot_total(run_ids: Sequence[int] | None = None) -> Optional[int]:
    if run_ids:
        run_id_tuple = tuple(run_ids)
        placeholders = ", ".join(f":run_{idx}" for idx in range(len(run_id_tuple)))
        query = f"SELECT COUNT(*) AS total FROM app_snapshots WHERE run_id IN ({placeholders})"
        params = {f"run_{idx}": run_id for idx, run_id in enumerate(run_id_tuple)}
    else:
        query = "SELECT COUNT(*) AS total FROM app_snapshots"
        params = None
    df = fetch_dataframe(query, params)
    if df.empty:
        return None
    return int(df.iloc[0]["total"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_neighbors(model: str = DEFAULT_EMBEDDING_MODEL) -> Dict[Tuple[int, int], List[dict]]:
    query = """
        SELECT
            n.run_id,
            n.track_id,
            n.neighbor_run_id,
            n.neighbor_track_id,
            n.similarity,
            n.rank,
            s.name AS neighbor_name,
            s.primary_genre_name AS neighbor_category,
            s.price AS neighbor_price,
            s.currency AS neighbor_currency,
            s.is_free AS neighbor_is_free,
            s.success_score AS neighbor_success_score,
            s.build_time_estimate AS neighbor_build_time,
            s.user_rating_count AS neighbor_rating_count,
            s.average_user_rating AS neighbor_review_score
        FROM app_snapshot_neighbors n
        JOIN app_snapshots s
          ON s.run_id = n.neighbor_run_id
         AND s.track_id = n.neighbor_track_id
        WHERE n.model = :model
        ORDER BY n.run_id DESC, n.track_id, n.rank
    """
    df = fetch_dataframe(query, {"model": model})
    if df.empty:
        return {}

    df["neighbor_price_label"] = np.where(
        df["neighbor_is_free"].fillna(False).astype(bool) | df["neighbor_price"].fillna(0).eq(0),
        "Free",
        df["neighbor_price"].fillna(0).map(lambda p: f"${p:.2f}"),
    )
    df["neighbor_similarity_pct"] = (df["similarity"] * 100).round(1)

    neighbor_map: Dict[Tuple[int, int], List[dict]] = {}
    grouped = df.groupby(["run_id", "track_id"], sort=False)
    for (run_id, track_id), group in grouped:
        neighbor_map[(run_id, track_id)] = [
            {
                "neighbor_run_id": row["neighbor_run_id"],
                "neighbor_track_id": row["neighbor_track_id"],
                "name": row["neighbor_name"],
                "category": row["neighbor_category"],
                "price_label": row["neighbor_price_label"],
                "similarity": float(row["similarity"]),
                "similarity_pct": float(row["neighbor_similarity_pct"]),
                "rank": int(row["rank"]),
                "success_score": row["neighbor_success_score"],
                "build_time_estimate": row["neighbor_build_time"],
                "rating_count": row["neighbor_rating_count"],
                "review_score": row["neighbor_review_score"],
            }
            for _, row in group.iterrows()
        ]
    return neighbor_map


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_cluster_data(model: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters_df = fetch_dataframe(
        """
            SELECT id, scope, model, label, keywords_json, size, avg_success, avg_build, avg_demand, created_at
            FROM app_snapshot_clusters
            WHERE model = :model
            ORDER BY size DESC
        """,
        {"model": model},
    )
    members_df = fetch_dataframe(
        """
            SELECT
                m.cluster_id,
                m.run_id,
                m.track_id,
                m.distance,
                s.name,
                s.primary_genre_name AS category,
                s.price,
                s.currency,
                s.is_free,
                s.success_score,
                s.build_time_estimate,
                s.user_rating_count,
                s.average_user_rating
            FROM app_snapshot_cluster_members AS m
            JOIN app_snapshot_clusters AS c
              ON c.id = m.cluster_id
            JOIN app_snapshots AS s
              ON s.run_id = m.run_id
             AND s.track_id = m.track_id
            WHERE c.model = :model
        """,
        {"model": model},
    )

    if clusters_df.empty or members_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    clusters_df["keywords"] = clusters_df["keywords_json"].apply(
        lambda val: json.loads(val) if isinstance(val, str) else (val or [])
    )
    clusters_df["label_display"] = clusters_df.apply(
        lambda row: f"{row['label']} ({row['size']})" if row.get("label") else f"Cluster {row['id']} ({row['size']})",
        axis=1,
    )
    clusters_df["avg_success_display"] = clusters_df["avg_success"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
    clusters_df["avg_build_display"] = clusters_df["avg_build"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
    clusters_df["avg_demand_display"] = clusters_df["avg_demand"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "–")

    members_df["price_label"] = np.where(
        members_df["is_free"].fillna(False).astype(bool) | members_df["price"].fillna(0).eq(0),
        "Free",
        members_df["price"].fillna(0).map(lambda p: f"${p:.2f}"),
    )
    members_df["distance"] = members_df["distance"].astype(float)
    members_df["similarity"] = 1 - members_df["distance"].clip(lower=0)
    members_df["similarity_pct"] = (members_df["similarity"] * 100).round(1)

    return clusters_df, members_df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_feature_table() -> Optional[pd.DataFrame]:
    try:
        df = fetch_dataframe("SELECT * FROM app_snapshot_deltas")
    except Exception:
        return None
    if df.empty:
        return None
    if "run_created_at" in df.columns:
        df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    if "is_new_track" in df.columns:
        df["is_new_track"] = df["is_new_track"].astype(bool)
    if "price_changed" in df.columns:
        df["price_changed"] = df["price_changed"].astype(bool)
    return df


# The remainder of the file can reuse the UI/rendering logic from streamlit_app_cloud.py
# ------------------------------------------------------------------------------
CATEGORY_HELP = "Use filters on the left to scope to categories, price tiers, and runs."


def render_sidebar(df: pd.DataFrame, run_catalog: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    categories = sorted(df["category_clean"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories,
    )

    price_tiers = sorted(df["price_tier"].unique())
    selected_tiers = st.sidebar.multiselect(
        "Price tier",
        options=price_tiers,
        default=price_tiers,
    )

    if run_catalog.empty:
        available_runs = pd.DataFrame(columns=["run_id", "run_date"])
    else:
        available_runs = run_catalog.copy()
    run_dates = sorted(available_runs["run_date"].dropna().unique(), reverse=True)
    show_all_dates = st.sidebar.checkbox("Show all scrape dates", value=not run_dates)
    if not show_all_dates and run_dates:
        default_date = run_dates[0]
        selected_date = st.sidebar.date_input(
            "Scrape date", value=default_date, min_value=min(run_dates), max_value=max(run_dates)
        )
        if isinstance(selected_date, list):
            selected_dates = selected_date
        else:
            selected_dates = [selected_date]
        selected_runs = (
            available_runs[available_runs["run_date"].isin(selected_dates)]["run_id"]
            .unique()
            .tolist()
        )
    else:
        selected_runs = available_runs["run_id"].unique().tolist()

    min_ratings = int(df["number_of_ratings"].fillna(0).min())
    max_ratings = int(df["number_of_ratings"].fillna(0).max())
    ratings_range = st.sidebar.slider(
        "User rating count",
        min_value=min_ratings,
        max_value=max_ratings,
        value=(min_ratings, max_ratings),
    )

    build_range = st.sidebar.slider(
        "Estimated build time (weeks)",
        min_value=float(df["build_time_estimate"].min()),
        max_value=float(df["build_time_estimate"].max()),
        value=(float(df["build_time_estimate"].min()), float(df["build_time_estimate"].max())),
    )

    success_range = st.sidebar.slider(
        "Success score",
        min_value=float(df["success_score"].min()),
        max_value=float(df["success_score"].max()),
        value=(float(df["success_score"].min()), float(df["success_score"].max())),
    )

    quick_win_only = st.sidebar.toggle(
        "Quick wins only (≤12 weeks & ≥70 score)",
        value=False,
    )

    return {
        "categories": selected_categories,
        "price_tiers": selected_tiers,
        "run_ids": selected_runs,
        "ratings_range": ratings_range,
        "build_range": build_range,
        "success_range": success_range,
        "quick_win_only": quick_win_only,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if filters["run_ids"]:
        filtered = df[df["run_id"].isin(filters["run_ids"])]
    else:
        filtered = df.copy()

    filtered = filtered[
        filtered["category_clean"].isin(filters["categories"])
        & filtered["price_tier"].isin(filters["price_tiers"])
    ]
    low_ratings, high_ratings = filters["ratings_range"]
    filtered = filtered[
        filtered["number_of_ratings"].fillna(0).between(low_ratings, high_ratings)
    ]
    low_build, high_build = filters["build_range"]
    filtered = filtered[
        filtered["build_time_estimate"].between(low_build, high_build)
    ]
    low_success, high_success = filters["success_range"]
    filtered = filtered[
        filtered["success_score"].between(low_success, high_success)
    ]
    if filters["quick_win_only"]:
        filtered = filtered[filtered["quick_win"]]
    return filtered


def plot_scatter(
    df: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    color_field: str,
    size_field: str,
) -> px.scatter:
    color_map = {
        category: px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)]
        for idx, category in enumerate(sorted(df["category_segment"].unique()))
    }
    fig = px.scatter(
        df,
        x=x_field,
        y=y_field,
        color=color_field,
        size=size_field,
        hover_name="name",
        hover_data={
            "developer": True,
            "category_clean": True,
            "price_tier": True,
            "price": ":.2f",
            "success_per_week": ":.2f",
            "number_of_ratings": True,
            "review_score": ":.2f",
        },
        color_discrete_map=color_map if color_field == "category_segment" else None,
        height=700,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.3)"), opacity=0.75))
    fig.update_layout(
        title=f"{AXIS_OPTIONS.get(y_field, y_field)} vs {AXIS_OPTIONS.get(x_field, x_field)}",
        xaxis_title=AXIS_OPTIONS.get(x_field, x_field),
        yaxis_title=AXIS_OPTIONS.get(y_field, y_field),
        legend_title="Category (price tier)" if color_field == "category_segment" else color_field.replace("_", " ").title(),
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    if x_field == "build_time_estimate" and y_field == "success_score":
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


# Remaining rendering helpers (3D scatter, summary plots, tables) match the
# cloud version. Reuse them to keep behaviour consistent.
plot_scatter_3d = cloud_ui.plot_scatter_3d
plot_category_summary = cloud_ui.plot_category_summary
plot_success_distribution = cloud_ui.plot_success_distribution
render_top_table = cloud_ui.render_top_table
render_opportunity_finder = cloud_ui.render_opportunity_finder
render_similarity_clusters = cloud_ui.render_similarity_clusters


def main() -> None:
    st.set_page_config(
        page_title="App Store Opportunity Explorer (Neon)",
        page_icon="📈",
        layout="wide",
    )
    st.title("App Store Opportunity Explorer – Neon Edition")
    st.caption("Backed by the Neon-hosted PostgreSQL database.")

    try:
        df = load_data()
    except RuntimeError as err:
        st.error(str(err))
        st.stop()
    except Exception as err:  # noqa: BLE001
        st.exception(err)
        st.stop()

    run_catalog = load_run_catalog()
    filters = render_sidebar(df, run_catalog)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No apps match the current filter combination. Charts may appear blank.")

    total_snapshots = load_snapshot_total(tuple(filters["run_ids"]))
    render_metrics(filtered_df, total_snapshots)

    feature_df = load_feature_table()
    neighbor_lookup = load_neighbors()
    clusters_df, members_df = load_cluster_data()

    tab_labels = [
        "Scatter plot",
        "3D scatter",
        "Category summary",
        "Success distribution",
        "Quick-win table",
        "Similarity clusters",
        "Opportunity finder",
    ]
    if feature_df is not None:
        tab_labels.append("Deltas")

    tabs = st.tabs(tab_labels)
    tab_scatter = tabs[0]
    tab_scatter_3d = tabs[1]
    tab_summary = tabs[2]
    tab_distribution = tabs[3]
    tab_table = tabs[4]
    tab_clusters = tabs[5]
    tab_opportunities = tabs[6]
    tab_deltas = tabs[7] if feature_df is not None else None

    with tab_scatter:
        controls_col, chart_col = st.columns([1, 3])
        with controls_col:
            st.caption("Map different metrics to each axis/encoding.")
            x_field = st.selectbox(
                "X-axis",
                options=list(AXIS_OPTIONS.keys()),
                index=list(AXIS_OPTIONS.keys()).index("build_time_estimate"),
                format_func=lambda key: AXIS_OPTIONS[key],
                key="scatter_x_axis",
            )
            y_field = st.selectbox(
                "Y-axis",
                options=list(AXIS_OPTIONS.keys()),
                index=list(AXIS_OPTIONS.keys()).index("success_score"),
                format_func=lambda key: AXIS_OPTIONS[key],
                key="scatter_y_axis",
            )
            color_field = st.selectbox(
                "Colour by",
                options=["category_segment", "price_tier"],
                index=0,
                format_func=lambda key: "Category" if key == "category_segment" else "Price tier",
                key="scatter_color_field",
            )
            size_field = st.selectbox(
                "Bubble size",
                options=["bubble_size", "number_of_ratings", "success_per_week"],
                index=0,
                format_func=lambda key: {
                    "bubble_size": "Confidence (log ratings)",
                    "number_of_ratings": "Rating count",
                    "success_per_week": "Success per week",
                }[key],
                key="scatter_size_field",
            )
        with chart_col:
            fig = plot_scatter(
                filtered_df,
                x_field=x_field,
                y_field=y_field,
                color_field=color_field,
                size_field=size_field,
            )
            st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption(
            "Use the legend and lasso/box select tools to inspect specific cohorts."
        )
        st.caption("This scatter compares effort vs. success for the filtered apps; adjust controls to explore variations.")

    with tab_scatter_3d:
        fig_3d = plot_scatter_3d(
            filtered_df,
            z_axis="success_per_week",
            color_field="category_segment",
        )
        st.plotly_chart(fig_3d, config={"displayModeBar": False})

    with tab_summary:
        summary_fig = plot_category_summary(filtered_df)
        st.plotly_chart(summary_fig, config={"displayModeBar": False})

    with tab_distribution:
        dist_fig = plot_success_distribution(filtered_df)
        st.plotly_chart(dist_fig, config={"displayModeBar": False})

    with tab_table:
        render_top_table(filtered_df)

    with tab_clusters:
        render_similarity_clusters(filtered_df, clusters_df, members_df)

    with tab_opportunities:
        render_opportunity_finder(filtered_df, neighbor_lookup)

    if tab_deltas is not None and feature_df is not None:
        with tab_deltas:
            render_deltas_tab(feature_df, filters)


def render_metrics(df: pd.DataFrame, total_available: int | None) -> None:
    total_apps = len(df)
    display_total = f"{total_apps:,}"
    if total_available and total_available >= 0:
        display_total = f"{total_apps:,} / {total_available:,}"

    quick_wins = int(df["quick_win"].sum()) if "quick_win" in df else 0
    avg_success = (
        df["success_score"].mean() if "success_score" in df and not df.empty else None
    )
    avg_build = (
        df["build_time_estimate"].mean()
        if "build_time_estimate" in df and not df.empty
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apps", display_total)
    col2.metric("Quick wins", f"{quick_wins:,}")
    col3.metric(
        "Average success",
        f"{avg_success:.1f}" if avg_success is not None else "–",
    )
    col4.metric(
        "Average build weeks",
        f"{avg_build:.1f}" if avg_build is not None else "–",
    )


def render_deltas_tab(feature_df: pd.DataFrame, filters: dict) -> None:
    st.subheader("Snapshot deltas (experimental)")
    st.caption("Comparing successive runs per app. Data sourced from app_snapshot_deltas.")
    st.caption("Track how success and rank metrics shift between runs. Use the metric toggle to focus on score or leaderboard movement.")

    delta_view = feature_df.copy()
    if filters["run_ids"]:
        delta_view = delta_view[delta_view["run_id"].isin(filters["run_ids"])]

    if delta_view.empty:
        st.info("No delta data available for the current run filters.")
        return

    metric_choice = st.selectbox(
        "Metric",
        options=["Success score", "Rank position"],
        index=0,
    )

    if metric_choice == "Success score":
        delta_col = "delta_success"
        delta_label = "Δ Success"
        run_title = "Average Δ success per run"
        run_ylabel = "Δ Success"
        improve_title = "Top improvers (success score)"
        decline_title = "Top decliners (success score)"
        category_title = "Category Δ success"
        improve_sort = lambda frame: frame.sort_values("delta_success", ascending=False).head(10)
        decline_sort = lambda frame: frame.sort_values("delta_success", ascending=True).head(10)
    else:
        delta_col = "delta_rank"
        delta_label = "Δ Rank"
        run_title = "Average Δ rank per run"
        run_ylabel = "Δ Rank"
        improve_title = "Biggest rank climbers"
        decline_title = "Biggest rank drops"
        category_title = "Category Δ rank"
        improve_sort = lambda frame: frame.sort_values("delta_rank", ascending=True).head(10)
        decline_sort = lambda frame: frame.sort_values("delta_rank", ascending=False).head(10)

    comparables = delta_view.copy()
    comparables[delta_label] = comparables[delta_col]

    run_summary = (
        comparables.groupby(["run_id", "run_created_at"], as_index=False)
        .agg(
            avg_delta_metric=(delta_col, "mean"),
            avg_delta_build=("delta_build_time", "mean"),
            returning_apps=("track_id", "count"),
        )
        .sort_values("run_created_at")
    )
    if not run_summary.empty:
        run_summary["run_created_at"] = pd.to_datetime(
            run_summary["run_created_at"], errors="coerce"
        )
        run_summary = run_summary.dropna(subset=["avg_delta_metric"])

    run_new = (
        delta_view.groupby(["run_id", "run_created_at"], as_index=False)
        .agg(new_apps=("is_new_track", "sum"))
    )
    if not run_new.empty:
        run_new["run_created_at"] = pd.to_datetime(run_new["run_created_at"], errors="coerce")
        run_new = run_new.dropna(subset=["new_apps"])

    col_line, col_bar = st.columns(2)
    if not run_summary.empty and run_summary["run_created_at"].notna().any():
        fig_delta = px.line(
            run_summary,
            x="run_created_at",
            y="avg_delta_metric",
            markers=True,
            title=run_title,
        )
        fig_delta.update_layout(xaxis_title="Run", yaxis_title=run_ylabel)
        col_line.plotly_chart(fig_delta, config={"displayModeBar": False})
    elif not run_summary.empty:
        fig_delta = px.line(
            run_summary,
            x="run_id",
            y="avg_delta_metric",
            markers=True,
            title=f"{run_title} (by run id)",
        )
        fig_delta.update_layout(xaxis_title="Run ID", yaxis_title=run_ylabel)
        col_line.plotly_chart(fig_delta, config={"displayModeBar": False})
    else:
        col_line.info("Not enough data to compute average deltas yet.")

    if not run_new.empty and run_new["run_created_at"].notna().any():
        fig_new = px.bar(
            run_new,
            x="run_created_at",
            y="new_apps",
            title="New apps per run",
        )
        fig_new.update_layout(xaxis_title="Run", yaxis_title="# New apps")
        col_bar.plotly_chart(fig_new, config={"displayModeBar": False})
    elif not run_new.empty:
        fig_new = px.bar(
            run_new,
            x="run_id",
            y="new_apps",
            title="New apps per run (by run id)",
        )
        fig_new.update_layout(xaxis_title="Run ID", yaxis_title="# New apps")
        col_bar.plotly_chart(fig_new, config={"displayModeBar": False})
    else:
        col_bar.info("No run data to display new app counts.")

    top_gain = improve_sort(comparables.dropna(subset=[delta_col]))
    top_loss = decline_sort(comparables.dropna(subset=[delta_col]))

    movers_cols = ["run_id", "track_id", "name", "run_created_at"]
    if metric_choice == "Rank position":
        movers_cols.extend(["best_rank", "prev_rank", "delta_rank"])
    else:
        movers_cols.extend(["success_score", "prev_success_score", "delta_success"])
    movers_cols.append("days_since_prev")

    col_gain, col_loss = st.columns(2)
    col_gain.markdown(f"**{improve_title}**")
    col_gain.dataframe(top_gain[movers_cols] if not top_gain.empty else pd.DataFrame(columns=movers_cols), width="stretch", hide_index=True)
    col_loss.markdown(f"**{decline_title}**")
    col_loss.dataframe(top_loss[movers_cols] if not top_loss.empty else pd.DataFrame(columns=movers_cols), width="stretch", hide_index=True)

    cat_summary = (
        comparables.dropna(subset=[delta_col])
        .groupby("category", as_index=False)
        .agg(avg_delta_metric=(delta_col, "mean"), apps=("track_id", "count"))
        .sort_values("avg_delta_metric", ascending=True if metric_choice == "Rank position" else False)
        .head(10)
    )
    if not cat_summary.empty:
        fig_cat = px.bar(
            cat_summary,
            x="category",
            y="avg_delta_metric",
            text="apps",
            title=category_title,
        )
        fig_cat.update_layout(xaxis_title="Category", yaxis_title=run_ylabel)
        st.plotly_chart(fig_cat, config={"displayModeBar": False})

    st.markdown("**Raw delta table**")
    show_columns = [
        "run_id",
        "track_id",
        "name",
        "run_created_at",
        "success_score",
        "prev_success_score",
        "delta_success",
        "build_time_estimate",
        "prev_build_time",
        "delta_build_time",
        "average_user_rating",
        "prev_rating",
        "delta_rating",
        "user_rating_count",
        "prev_rating_count",
        "delta_rating_count",
        "days_since_prev",
        "price",
        "prev_price",
        "price_changed",
        "best_rank",
        "prev_rank",
        "delta_rank",
        "is_new_track",
    ]
    available_columns = [col for col in show_columns if col in delta_view.columns]
    st.dataframe(
        delta_view[available_columns]
        .sort_values(["track_id", "run_id"], ascending=[True, False]),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
