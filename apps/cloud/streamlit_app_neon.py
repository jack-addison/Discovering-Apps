#!/usr/bin/env python3
"""Neon Streamlit app with cluster browsing support."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEON_DATABASE_URL = os.environ.get("PROTOTYPE_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
CACHE_TTL_SECONDS = 300


@st.cache_resource
def get_engine() -> Engine:
    if not NEON_DATABASE_URL:
        raise RuntimeError(
            "Set PROTOTYPE_DATABASE_URL (or NEON_DATABASE_URL) before launching the app."
        )
    dsn = NEON_DATABASE_URL
    if dsn.startswith("postgres://"):
        dsn = dsn.replace("postgres://", "postgresql+psycopg://", 1)
    elif dsn.startswith("postgresql://") and "+" not in dsn:
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(dsn, pool_pre_ping=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_scopes() -> List[str]:
    query = "SELECT DISTINCT scope FROM app_snapshot_clusters ORDER BY scope"
    with get_engine().connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    return [row[0] for row in rows] or []


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_cluster_data(scope: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_engine()
    scope_filter = "WHERE scope = :scope" if scope else ""
    params: Dict[str, Any] = {"scope": scope} if scope else {}
    query_clusters = f"""
        SELECT id, scope, model, label, keywords_json, size, avg_success, avg_build, avg_demand, created_at
        FROM app_snapshot_clusters
        {scope_filter}
        ORDER BY created_at DESC, size DESC
    """
    query_members = """
        SELECT
            m.cluster_id,
            s.run_id,
            s.track_id,
            s.name,
            s.primary_genre_name AS category,
            s.price,
            s.currency,
            s.is_free,
            s.average_user_rating,
            s.user_rating_count,
            s.success_score,
            s.build_time_estimate,
            m.distance
        FROM app_snapshot_cluster_members m
        JOIN app_snapshots s
          ON s.run_id = m.run_id
         AND s.track_id = m.track_id
        WHERE EXISTS (
            SELECT 1 FROM app_snapshot_clusters c
            WHERE c.id = m.cluster_id
            """ + ("AND c.scope = :scope" if scope else "") + """
        )
    """
    with engine.connect() as conn:
        clusters_df = pd.read_sql_query(text(query_clusters), conn, params=params)
        members_df = pd.read_sql_query(text(query_members), conn, params=params)
        descriptions = pd.read_sql_query(text("SELECT run_id, track_id, description FROM app_snapshots"), conn)
    members_df = members_df.merge(descriptions, on=["run_id", "track_id"], how="left")
    if clusters_df.empty or members_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    clusters_df["keywords"] = clusters_df["keywords_json"].apply(
        lambda val: json.loads(val) if isinstance(val, str) else (val or [])
    )
    clusters_df["label_display"] = clusters_df.apply(
        lambda row: f"{row['label']} ({row['size']})" if row.get("label") else f"Cluster {row['id']} ({row['size']})",
        axis=1,
    )
    clusters_df["avg_success_display"] = clusters_df["avg_success"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "–")
    clusters_df["avg_build_display"] = clusters_df["avg_build"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
    clusters_df["avg_demand_display"] = clusters_df["avg_demand"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "–")

    members_df["price_label"] = members_df["price"].fillna(0).map(lambda p: "Free" if p == 0 else f"${p:.2f}")
    members_df["distance"] = members_df["distance"].astype(float)
    members_df["similarity_pct"] = (1 - members_df["distance"]).clip(lower=0).round(3) * 100
    return clusters_df, members_df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_app_names() -> List[str]:
    query = """
        SELECT DISTINCT name
        FROM app_snapshots
        WHERE name IS NOT NULL
        ORDER BY name ASC
    """
    with get_engine().connect() as conn:
        rows = conn.execute(text(query)).fetchall()
    return [row[0] for row in rows]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_deltas_data() -> pd.DataFrame:
    query = "SELECT * FROM app_snapshot_deltas"
    with get_engine().connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    if df.empty:
        return df
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    df["prev_run_created_at"] = pd.to_datetime(df["prev_run_created_at"], errors="coerce")
    df["is_new_track"] = df["is_new_track"].fillna(False).astype(bool)
    df["price_changed"] = df["price_changed"].fillna(False).astype(bool)
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_dissatisfied_counts() -> pd.DataFrame:
    query = """
        SELECT
            d.run_id,
            COALESCE(sr.created_at, MAX(s.scraped_at)) AS run_created_at,
            COUNT(*) AS flagged
        FROM app_snapshot_dissatisfied d
        LEFT JOIN scrape_runs sr ON sr.id = d.run_id
        LEFT JOIN app_snapshots s
          ON s.run_id = d.run_id
         AND s.track_id = d.track_id
        GROUP BY d.run_id, sr.created_at
        ORDER BY d.run_id
    """
    with get_engine().connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    if df.empty:
        return df
    df["run_created_at"] = pd.to_datetime(df["run_created_at"], errors="coerce")
    return df


def render_clusters_tab() -> None:
    scopes = load_scopes()
    if not scopes:
        st.info("No clusters found. Generate embeddings and run the clustering job first.")
        return

    selected_scope = st.selectbox("Cluster scope", scopes, index=0)
    clusters_df, members_df = load_cluster_data(selected_scope)
    if clusters_df.empty or members_df.empty:
        st.info("No cluster data for the selected scope.")
        return

    overview = clusters_df[
        ["id", "label_display", "size", "avg_success_display", "avg_build_display", "avg_demand_display", "keywords"]
    ].copy()
    overview = overview.rename(
        columns={
            "label_display": "Cluster",
            "size": "Apps",
            "avg_success_display": "Avg success",
            "avg_build_display": "Avg build weeks",
            "avg_demand_display": "Avg demand",
            "keywords": "Keywords",
        }
    )
    overview.insert(0, "Inspect", False)

    edited = st.data_editor(
        overview.head(200),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Inspect": st.column_config.CheckboxColumn(
                help="Check a cluster to preview its members below.",
                default=False,
            )
        },
        key="neon_cluster_overview",
    )

    selected_ids = edited.loc[edited["Inspect"], "id"].tolist()
    if selected_ids:
        cluster_id = selected_ids[0]
    else:
        cluster_id = clusters_df.iloc[0]["id"]

    current_cluster = clusters_df.loc[clusters_df["id"] == cluster_id].iloc[0]
    keywords = ", ".join(current_cluster.get("keywords", [])) or "Unlabelled"

    metric_cols = st.columns(2)
    metric_cols[0].metric("Apps", int(current_cluster["size"]))
    metric_cols[1].metric("Avg rating", current_cluster["avg_success_display"])
    st.caption(f"Keywords: {keywords}")

    member_rows = members_df[members_df["cluster_id"] == cluster_id].copy()

    st.divider()
    st.markdown("#### Opportunity filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    max_rating = float(member_rows["average_user_rating"].max(skipna=True) or 5.0)
    rating_threshold = filter_col1.slider(
        "Max average rating",
        min_value=1.0,
        max_value=5.0,
        value=min(3.0, max_rating),
        step=0.1,
        help="Only show apps whose average rating is below this value.",
    )
    default_min_ratings = int(member_rows["user_rating_count"].mean() or 0)
    min_ratings = filter_col2.number_input(
        "Min rating count",
        min_value=0,
        value=default_min_ratings,
        step=10,
        help="Ignore low-volume apps by requiring at least this many ratings.",
    )
    price_options = ["Free", "Paid"]
    selected_prices = filter_col3.multiselect(
        "Price tier",
        options=price_options,
        default=price_options,
    )

    filtered = member_rows.copy()
    filtered = filtered[
        (filtered["average_user_rating"].fillna(5.0) <= rating_threshold)
        & (filtered["user_rating_count"].fillna(0) >= min_ratings)
        & (filtered["price_label"].apply(lambda label: ("Free" if label == "Free" else "Paid") in selected_prices))
    ]

    filtered["rating_gap"] = (rating_threshold - filtered["average_user_rating"]).clip(lower=0)
    filtered["opportunity_score"] = filtered["user_rating_count"].fillna(0) * filtered["rating_gap"]
    member_rows["rating_gap"] = (rating_threshold - member_rows["average_user_rating"]).clip(lower=0)
    member_rows["opportunity_score"] = member_rows["user_rating_count"].fillna(0) * member_rows["rating_gap"]
    st.caption(
        "Opportunity score = rating count × (max rating threshold − average rating)."
        " Higher scores highlight popular apps with low satisfaction."
    )
    filtered.sort_values(["opportunity_score", "user_rating_count"], ascending=False, inplace=True)

    top_opportunities = filtered.head(20)[
        [
            "name",
            "category",
            "price_label",
            "average_user_rating",
            "user_rating_count",
            "opportunity_score",
            "description",
        ]
    ].rename(
        columns={
            "name": "App",
            "category": "Category",
            "price_label": "Price",
            "average_user_rating": "Avg rating",
            "user_rating_count": "Rating count",
            "opportunity_score": "Opportunity score",
            "description": "Description",
        }
    )

    st.markdown("#### Top opportunities")
    st.dataframe(
        top_opportunities,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Avg rating": st.column_config.NumberColumn(format="%.2f"),
            "Rating count": st.column_config.NumberColumn(format="%d"),
            "Opportunity score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown("#### Cluster members (full list)")
    member_rows.sort_values("similarity_pct", ascending=False, inplace=True)
    display = member_rows[
        [
            "name",
            "category",
            "price_label",
            "average_user_rating",
            "user_rating_count",
            "similarity_pct",
            "opportunity_score",
            "description",
        ]
    ].rename(
        columns={
            "name": "App",
            "category": "Category",
            "price_label": "Price",
            "average_user_rating": "Avg rating",
            "user_rating_count": "Rating count",
            "similarity_pct": "Similarity %",
            "opportunity_score": "Opportunity score",
            "description": "Description",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Avg rating": st.column_config.NumberColumn(format="%.2f"),
            "Rating count": st.column_config.NumberColumn(format="%d"),
            "Similarity %": st.column_config.NumberColumn(format="%.1f"),
            "Opportunity score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_apps_tab() -> None:
    app_names = load_app_names()
    if not app_names:
        st.info("No apps found. Run the scraper first.")
        return
    st.selectbox("All apps", options=app_names, index=0, label_visibility="visible")


def render_deltas_tab() -> None:
    deltas_df = load_deltas_data()
    if deltas_df.empty:
        st.info("No delta data available. Run the Neon delta builder script first.")
        return

    run_options = sorted(deltas_df["run_id"].unique(), reverse=True)
    default_runs = run_options[: min(5, len(run_options))]
    selected_runs = st.multiselect(
        "Filter runs",
        options=run_options,
        default=default_runs,
        help="Choose specific run IDs to focus on recent changes.",
    )
    if selected_runs:
        view = deltas_df[deltas_df["run_id"].isin(selected_runs)].copy()
    else:
        view = deltas_df.copy()

    if view.empty:
        st.warning("No delta rows match the selected filters.")
        return

    metric_config = {
        "Rating average": {
            "col": "delta_rating",
            "label": "Δ rating",
            "run_title": "Average Δ rating by run",
            "run_ylabel": "Avg Δ rating",
            "category_title": "Top categories by avg Δ rating",
            "improve_title": "Top positive movers (rating)",
            "decline_title": "Top negative movers (rating)",
            "improve_sort": lambda df: df.nlargest(10, "delta_rating"),
            "decline_sort": lambda df: df.nsmallest(10, "delta_rating"),
            "note": "Positive values indicate improving user ratings.",
        },
        "Rating count": {
            "col": "delta_rating_count",
            "label": "Δ rating count",
            "run_title": "Average Δ rating count by run",
            "run_ylabel": "Avg Δ rating count",
            "category_title": "Top categories by avg Δ rating count",
            "improve_title": "Top positive movers (rating volume)",
            "decline_title": "Top negative movers (rating volume)",
            "improve_sort": lambda df: df.nlargest(10, "delta_rating_count"),
            "decline_sort": lambda df: df.nsmallest(10, "delta_rating_count"),
            "note": "Positive values mean the app accumulated more ratings since the prior run.",
        },
    }

    metric_choice = st.radio(
        "Metric",
        options=list(metric_config.keys()),
        index=0,
        horizontal=True,
    )
    config = metric_config[metric_choice]
    delta_col = config["col"]

    comparables = view[~view["is_new_track"]].copy()
    col_a, col_b, col_c, col_d = st.columns(4)
    avg_delta = comparables[delta_col].dropna().mean()
    col_a.metric(config["label"], f"{avg_delta:+.3f}" if pd.notna(avg_delta) else "--", help=config["note"])
    avg_rating_delta = comparables["delta_rating"].dropna().mean()
    col_b.metric("Avg Δ rating (all)", f"{avg_rating_delta:+.3f}" if pd.notna(avg_rating_delta) else "--")
    col_c.metric("New apps this window", int(view["is_new_track"].sum()))
    median_gap = comparables["days_since_prev"].dropna().median()
    col_d.metric("Median days between runs", f"{median_gap:.1f}" if pd.notna(median_gap) else "--")

    comparables["run_date"] = comparables["run_created_at"].dt.date
    run_summary = (
        comparables.dropna(subset=[delta_col])
        .groupby(["run_date"], as_index=False)
        .agg(avg_delta=(delta_col, "mean"), returning_apps=("track_id", "count"))
        .sort_values("run_date")
    )
    if not run_summary.empty:
        fig_runs = px.line(
            run_summary,
            x="run_date",
            y="avg_delta",
            markers=True,
            title=config["run_title"],
        )
        fig_runs.update_layout(xaxis_title="Calendar date", yaxis_title=config["run_ylabel"])
        st.plotly_chart(fig_runs, use_container_width=True, config={"displayModeBar": False})

    cat_summary = (
        comparables.dropna(subset=[delta_col])
        .groupby("category", as_index=False)
        .agg(avg_delta=(delta_col, "mean"), apps=("track_id", "count"))
        .sort_values("avg_delta", ascending=False)
    )
    if not cat_summary.empty:
        fig_cat = px.bar(
            cat_summary.head(10),
            x="category",
            y="avg_delta",
            text="apps",
            title=config["category_title"],
        )
        fig_cat.update_layout(xaxis_title="Category", yaxis_title=config["run_ylabel"])
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

    col_gain, col_loss = st.columns(2)
    top_gain = config["improve_sort"](comparables).copy()
    col_gain.markdown(f"**{config['improve_title']}**")
    col_gain.dataframe(
        top_gain[
            ["run_id", "track_id", "name", "category", delta_col, "average_user_rating", "user_rating_count"]
        ].rename(columns={delta_col: config["label"]}),
        hide_index=True,
        use_container_width=True,
    )
    top_loss = config["decline_sort"](comparables).copy()
    col_loss.markdown(f"**{config['decline_title']}**")
    col_loss.dataframe(
        top_loss[
            ["run_id", "track_id", "name", "category", delta_col, "average_user_rating", "user_rating_count"]
        ].rename(columns={delta_col: config["label"]}),
        hide_index=True,
        use_container_width=True,
    )

    timeline_df = load_dissatisfied_counts()
    if not timeline_df.empty:
        timeline_df["run_date"] = timeline_df["run_created_at"].dt.date
        timeline_agg = (
            timeline_df.groupby("run_date", as_index=False)["flagged"].sum().sort_values("run_date")
        )
        fig_flagged = px.bar(
            timeline_agg,
            x="run_date",
            y="flagged",
            title="Flagged dissatisfied apps per run",
        )
        fig_flagged.update_layout(xaxis_title="Calendar date", yaxis_title="# dissatisfied apps")
        st.plotly_chart(fig_flagged, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Neon Opportunity Explorer", page_icon="📊", layout="wide")
    st.title("Neon Opportunity Explorer")
    st.caption("Clusters derived from Postgres + embeddings.")

    tab_clusters, tab_apps, tab_deltas = st.tabs(["Clusters", "Apps", "Deltas"])
    with tab_clusters:
        render_clusters_tab()
    with tab_apps:
        render_apps_tab()
    with tab_deltas:
        render_deltas_tab()


if __name__ == "__main__":
    main()
