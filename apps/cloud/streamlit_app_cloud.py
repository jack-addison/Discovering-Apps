#!/usr/bin/env python3
"""Streamlit dashboard for exploring Stage 2 success vs build time scores."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlitecloud

CONNECTION_URI = "sqlitecloud://cky1wteehz.g4.sqlite.cloud:8860/app_store_apps_v2.db?apikey=HuDMLrUJedgC54VQOfbfQ489AY0aDivJ6XXfe1r01Wo"
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


def get_connection():
    return sqlitecloud.connect(CONNECTION_URI)


def fetch_dataframe(query: str, params: Tuple[Any, ...] | List[Any] | None = None) -> pd.DataFrame:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params or [])
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


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
def load_neighbors(model: str = DEFAULT_EMBEDDING_MODEL) -> Dict[Tuple[int, int], List[dict]]:
    try:
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
                WHERE n.model = ?
                ORDER BY n.run_id DESC, n.track_id, n.rank
            """
        df = fetch_dataframe(query, params=(model,))
    except (Exception, pd.errors.DatabaseError):
        return {}
    if df.empty:
        return {}

    df["neighbor_price_label"] = np.where(
        df["neighbor_is_free"].fillna(0).astype(bool) | df["neighbor_price"].fillna(0).eq(0),
        "Free",
        df["neighbor_price"].fillna(0).map(lambda p: f"${p:.2f}"),
    )
    df["neighbor_similarity_pct"] = (df["similarity"] * 100).round(1)

    neighbor_map: Dict[Tuple[int, int], List[dict]] = {}
    for (run_id, track_id), group in df.groupby(["run_id", "track_id"]):
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
    try:
        clusters_df = fetch_dataframe(
            """
                SELECT id, scope, model, label, keywords_json, size, avg_success, avg_build, avg_demand, created_at
                FROM app_snapshot_clusters
                WHERE model = ?
                ORDER BY size DESC
                """,
            params=(model,),
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
                WHERE c.model = ?
                """,
            params=(model,),
        )
    except (Exception, pd.errors.DatabaseError):
        return pd.DataFrame(), pd.DataFrame()

    if not clusters_df.empty:
        clusters_df["keywords"] = clusters_df["keywords_json"].apply(lambda val: json.loads(val) if isinstance(val, str) else [])
        clusters_df["label_display"] = clusters_df.apply(
            lambda row: f"{row['label']} ({row['size']})" if row.get("label") else f"Cluster {row['id']} ({row['size']})",
            axis=1,
        )
        clusters_df["avg_success_display"] = clusters_df["avg_success"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
        clusters_df["avg_build_display"] = clusters_df["avg_build"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
        clusters_df["avg_demand_display"] = clusters_df["avg_demand"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "–")

    if not members_df.empty:
        members_df["price_label"] = np.where(
            members_df["is_free"].fillna(0).astype(bool) | members_df["price"].fillna(0).eq(0),
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
    except (Exception, pd.errors.DatabaseError):
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


def render_sidebar(df: pd.DataFrame) -> dict:
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

    run_dates = sorted(df["run_date"].dropna().unique(), reverse=True)
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
        selected_runs = df[df["run_date"].isin(selected_dates)]["run_id"].unique().tolist()
    else:
        selected_runs = df["run_id"].unique().tolist()

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


def plot_category_summary(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.groupby("category_segment")
        .agg(
            apps=("track_id", "count"),
            quick_wins=("quick_win", "sum"),
            median_success=("success_score", "median"),
            median_build=("build_time_estimate", "median"),
        )
        .sort_values("apps", ascending=False)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary.index,
            y=summary["apps"],
            name="Apps in view",
            marker_color="rgba(54, 162, 235, 0.7)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary.index,
            y=summary["quick_wins"],
            name="Quick wins",
            marker_color="rgba(75, 192, 75, 0.7)",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Apps vs quick wins by category",
        xaxis_title="Category (price tier)",
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=120),
    )
    return fig


def plot_success_distribution(df: pd.DataFrame) -> px.box:
    fig = px.box(
        df,
        x="price_tier",
        y="success_score",
        color="price_tier",
        points="suspectedoutliers",
        title="Success score distribution by price tier",
        template="plotly_white",
        labels={"success_score": "Success score", "price_tier": "Price tier"},
    )
    fig.update_traces(marker=dict(opacity=0.6))
    return fig


def plot_scatter_3d(df: pd.DataFrame, z_axis: str, color_field: str) -> px.scatter_3d:
    axis_labels = {
        "success_per_week": "Success per week",
        "number_of_ratings": "Rating count",
        "review_score": "Average review",
        "number_of_downloads": "Download proxy (ratings)",
    }
    fig = px.scatter_3d(
        df,
        x="build_time_estimate",
        y="success_score",
        z=z_axis,
        color=color_field,
        size="bubble_size",
        hover_name="name",
        hover_data={
            "developer": True,
            "category_segment": True,
            "price_tier": True,
            "success_per_week": ":.2f",
            "number_of_ratings": True,
            "review_score": ":.2f",
            "price": ":.2f",
        },
        template="plotly_white",
        height=700,
    )
    fig.update_layout(
        title=f"3D view: Success vs build time vs {axis_labels.get(z_axis, z_axis)}",
        scene=dict(
            xaxis_title="Build time (weeks)",
            yaxis_title="Success score",
            zaxis_title=axis_labels.get(z_axis, z_axis),
        ),
        legend_title=color_field.replace("_", " ").title(),
    )
    return fig


def render_metrics(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Apps shown", len(df))
    with col2:
        st.metric("Median build time", f"{df['build_time_estimate'].median():.1f} weeks")
    with col3:
        st.metric("Median success score", f"{df['success_score'].median():.1f}")

    run_labels = (
        df[["run_id", "run_created_at_str"]]
        .drop_duplicates()
        .sort_values(by="run_created_at_str", ascending=False)
        .apply(
            lambda row: f"{int(row.run_id)} ({row.run_created_at_str or 'unknown'})",
            axis=1,
        )
        .tolist()
    )
    if run_labels:
        st.caption("Runs shown: " + ", ".join(run_labels))
    st.caption(
        "Bubble size scales with log10(rating count + 1). Apply filters on the left to focus the chart."
    )


def render_top_table(df: pd.DataFrame) -> None:
    st.subheader("Quick-win candidates")
    top = (
        df[df["quick_win"]]
        .sort_values("review_score", ascending=False)
        .loc[
            :,
            [
                "name",
                "category_segment",
                "success_score",
                "build_time_estimate",
                "success_per_week",
                "review_score",
                "number_of_ratings",
                "price_tier",
                "price",
                "developer",
                "success_reasoning",
            ],
        ]
    )
    if top.empty:
        st.info("No apps currently meet the quick-win criteria under the selected filters.")
    else:
        st.dataframe(
            top.rename(
                columns={
                    "category_segment": "Category",
                    "success_score": "Success score",
                    "build_time_estimate": "Build weeks",
                    "success_per_week": "Score per week",
                    "review_score": "Avg review",
                    "number_of_ratings": "Ratings",
                    "price": "Price",
                    "success_reasoning": "Reasoning",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    st.caption("Shows the filtered quick-win apps along with Stage 2 scores, derivatives, and reasoning snippets.")


def render_similarity_clusters(
    filtered_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    members_df: pd.DataFrame,
) -> None:
    st.subheader("Similarity clusters")
    st.caption(
        "Clusters are generated from text embeddings (top keywords shown). Use them to explore competitive pods and compare apps with similar positioning."
    )

    if clusters_df.empty or members_df.empty:
        st.info(
            "No cluster data available. Run `python -m src.local.analysis.build_clusters` after generating embeddings to populate this view."
        )
        return

    top_clusters = clusters_df[[
        "id",
        "label_display",
        "size",
        "avg_success_display",
        "avg_build_display",
        "avg_demand_display",
        "keywords",
    ]].copy()
    top_clusters = top_clusters.rename(
        columns={
            "label_display": "Cluster",
            "size": "Apps",
            "avg_success_display": "Avg success",
            "avg_build_display": "Avg build weeks",
            "avg_demand_display": "Avg demand",
            "keywords": "Keywords",
        }
    )
    top_clusters.insert(0, "Inspect", False)

    edited_clusters = st.data_editor(
        top_clusters.head(200),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Inspect": st.column_config.CheckboxColumn(
                help="Check a cluster to preview its members below.",
                default=False,
            ),
        },
        key="cluster_overview_editor",
    )

    selected_ids = edited_clusters.loc[edited_clusters["Inspect"], "id"].tolist()
    if selected_ids:
        selected_cluster_id = selected_ids[0]
    else:
        selected_cluster_id = clusters_df.iloc[0]["id"]
    cluster_row = clusters_df.loc[clusters_df["id"] == selected_cluster_id].iloc[0]
    keywords = ", ".join(cluster_row.get("keywords", [])) or "Unlabelled"

    metric_cols = st.columns(4)
    metric_cols[0].metric("Apps", int(cluster_row["size"]))
    metric_cols[1].metric("Avg success", cluster_row.get("avg_success_display", "–"))
    metric_cols[2].metric("Avg build weeks", cluster_row.get("avg_build_display", "–"))
    metric_cols[3].metric("Avg demand", cluster_row.get("avg_demand_display", "–"))

    st.markdown(f"**Keywords:** {keywords}")
    st.caption(
        "Similarity measured via embedding cosine distance. Demand = rating volume × rating gap."
    )

    cluster_members = members_df[members_df["cluster_id"] == selected_cluster_id].copy()
    cluster_members.sort_values("similarity", ascending=False, inplace=True)
    filtered_keys = set(zip(filtered_df["run_id"], filtered_df["track_id"]))
    cluster_members["in_filters"] = cluster_members.apply(
        lambda row: (row["run_id"], row["track_id"]) in filtered_keys,
        axis=1,
    )

    st.markdown("#### Cluster members")
    member_display = cluster_members[[
        "name",
        "category",
        "price_label",
        "success_score",
        "build_time_estimate",
        "user_rating_count",
        "average_user_rating",
        "similarity_pct",
        "in_filters",
    ]].rename(
        columns={
            "name": "App",
            "category": "Category",
            "price_label": "Price",
            "success_score": "Success",
            "build_time_estimate": "Build weeks",
            "user_rating_count": "Rating count",
            "average_user_rating": "Avg rating",
            "similarity_pct": "Similarity %",
            "in_filters": "Matches current filters",
        }
    )
    st.dataframe(
        member_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Similarity %": st.column_config.NumberColumn(format="%.1f"),
            "Success": st.column_config.NumberColumn(format="%.0f"),
            "Build weeks": st.column_config.NumberColumn(format="%.1f"),
            "Avg rating": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    if cluster_members["in_filters"].any():
        st.caption(
            "Rows highlighted with `Matches current filters` = True align with the sidebar filters."
        )
    else:
        st.caption("Adjust sidebar filters to intersect with this cluster.")


def render_opportunity_finder(df: pd.DataFrame, neighbor_lookup: Dict[Tuple[int, int], List[dict]]) -> None:
    st.subheader("High-price, low-rating opportunities")
    st.info(
        "Work top to bottom: refine the cohort with the threshold controls, enforce an execution floor, then review the ranked tables to prioritise which incumbents to clone."
    )
    if df.empty:
        st.info("No data available under the current filters.")
        return

    working = df[
        df["price"].notna()
        & df["number_of_ratings"].notna()
        & df["review_score"].notna()
        & df["build_time_estimate"].notna()
    ].copy()

    if working.empty:
        st.info("No rows contain complete price, rating count, rating score, and build-time data.")
        return

    with st.expander("Audience & sentiment thresholds", expanded=True):
        st.caption(
            "Focus on premium incumbents with traction but unhappy users and a manageable build scope."
        )
        col_price, col_ratings, col_score, col_effort = st.columns(4)
        with col_price:
            price_pct = st.slider(
                "Price ≥ percentile",
                min_value=50,
                max_value=100,
                value=75,
                help="Select apps priced in the top percentile range.",
            )
        with col_ratings:
            ratings_pct = st.slider(
                "Rating count ≥ percentile",
                min_value=50,
                max_value=100,
                value=75,
                help="Select apps with rating counts in the top percentile range.",
            )
        with col_score:
            rating_cap = st.slider(
                "Average rating ≤",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                step=0.1,
                help="Upper bound for average review score.",
            )
        with col_effort:
            effort_cap = st.slider(
                "Build weeks ≤",
                min_value=1.0,
                max_value=40.0,
                value=12.0,
                step=0.5,
                help="Upper bound for Stage 2 build-time estimate.",
            )
        dissatisfaction_pct = st.slider(
            "Demand dissatisfaction ≥ percentile",
            min_value=50,
            max_value=100,
            value=75,
            help="Targets apps with a large, dissatisfied audience (rating count × rating gap).",
        )
        st.caption(
            "Demand dissatisfaction multiplies rating volume by the rating gap (5 − average rating). The percentile column shows how extreme the signal is within the loaded dataset."
        )

    with st.expander("Execution floor", expanded=False):
        risk_metric = st.selectbox(
            "Execution risk metric",
            options=["success_score", "success_per_week"],
            format_func=lambda key: "Stage 2 success score" if key == "success_score" else "Success per week",
            help="Apply a minimum bar for commercial upside: either the raw Stage 2 score or success divided by build time.",
        )
        risk_series = (
            working[risk_metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if risk_series.empty:
            st.warning("Unable to evaluate execution risk for the current dataset.")
            return

        risk_min = float(risk_series.min())
        risk_max = float(risk_series.max())
        if abs(risk_min - risk_max) < 1e-6:
            st.info(f"Execution risk metric is uniform at {risk_min:.1f}; using that value as the filter.")
            risk_threshold = risk_min
        else:
            default_threshold = risk_series.quantile(0.75)
            default_threshold = float(min(risk_max, max(risk_min, default_threshold)))
            risk_threshold = st.slider(
                "Minimum execution score",
                min_value=risk_min,
                max_value=risk_max,
                value=float(default_threshold),
                step=0.1 if risk_metric == "success_score" else 0.05,
                help="Raise this to focus on derivatives that also look likely to succeed.",
            )

    price_cutoff = working["price"].quantile(price_pct / 100) if not working["price"].empty else None
    ratings_cutoff = working["number_of_ratings"].quantile(ratings_pct / 100) if not working["number_of_ratings"].empty else None
    dissatisfaction_cutoff = (
        working["demand_dissatisfaction"].quantile(dissatisfaction_pct / 100)
        if not working["demand_dissatisfaction"].empty
        else None
    )

    if price_cutoff is None or ratings_cutoff is None or dissatisfaction_cutoff is None:
        st.warning("Unable to compute percentile cutoffs. Adjust filters or thresholds.")
        return

    filtered_opps = working[
        (working["price"] >= price_cutoff)
        & (working["number_of_ratings"] >= ratings_cutoff)
        & (working["review_score"] <= rating_cap)
        & (working["build_time_estimate"] <= effort_cap)
        & (working[risk_metric] >= risk_threshold)
        & (working["demand_dissatisfaction"] >= dissatisfaction_cutoff)
    ].copy()

    st.caption(
        f"Price ≥ {price_cutoff:.2f}, rating count ≥ {int(ratings_cutoff)}, rating ≤ {rating_cap:.2f}, build weeks ≤ {effort_cap:.1f}, "
        f"{'success score' if risk_metric == 'success_score' else 'success per week'} ≥ {risk_threshold:.1f}, "
        f"demand dissatisfaction ≥ {int(dissatisfaction_cutoff)}."
    )

    if filtered_opps.empty:
        st.info("No apps match the current criteria. Adjust the sliders to broaden the search.")
    else:
        sort_option = st.selectbox(
            "Sort results by",
            options=[
                "Highest price",
                "Highest rating count",
                "Lowest rating score",
                "Lowest build weeks",
                "Highest demand dissatisfaction",
            ],
        )
        if sort_option == "Highest price":
            filtered_opps.sort_values("price", ascending=False, inplace=True)
        elif sort_option == "Highest rating count":
            filtered_opps.sort_values("number_of_ratings", ascending=False, inplace=True)
        elif sort_option == "Lowest rating score":
            filtered_opps.sort_values("review_score", ascending=True, inplace=True)
        elif sort_option == "Highest demand dissatisfaction":
            filtered_opps.sort_values("demand_dissatisfaction", ascending=False, inplace=True)
        else:
            filtered_opps.sort_values("build_time_estimate", ascending=True, inplace=True)

        neighbor_lists = [
            neighbor_lookup.get((row["run_id"], row["track_id"]), [])
            for _, row in filtered_opps.iterrows()
        ]
        filtered_opps = filtered_opps.assign(neighbors=neighbor_lists)

        def format_similar(entries: List[dict], limit: int = 3) -> str:
            if not entries:
                return "—"
            snippets = [
                f"{item['name']} ({item['similarity']:.2f})"
                for item in entries[:limit]
            ]
            return ", ".join(snippets)

        filtered_opps = filtered_opps.assign(
            opportunity_note=filtered_opps.apply(
                lambda row: (
                    f"${row['price']:.2f} {row['category_segment']} app with "
                    f"{int(row['number_of_ratings']):,} ratings averaging {row['review_score']:.1f}/5; "
                    f"{row['build_time_estimate']:.1f} build weeks, success {row['success_score']:.0f}, "
                    f"demand dissatisfaction {row['demand_dissatisfaction']:.0f} ({row['demand_dissatisfaction_percentile']:.0f}th %tile)."
                ),
                axis=1,
            ),
            similar_apps=filtered_opps["neighbors"].apply(format_similar),
        )

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Matches", len(filtered_opps))
        col2.metric("Avg price", f"{filtered_opps['price'].mean():.2f}")
        col3.metric("Avg rating", f"{filtered_opps['review_score'].mean():.2f}")
        col4.metric(
            "Avg demand",
            f"{filtered_opps['demand_dissatisfaction'].mean():.0f}",
        )
        col5.metric(
            "Avg execution",
            f"{filtered_opps[risk_metric].mean():.1f}",
        )
        col6.metric(
            "Avg demand %tile",
            f"{filtered_opps['demand_dissatisfaction_percentile'].mean():.0f}",
        )

        display_columns = [
            "name",
            "category_segment",
            "price",
            "number_of_ratings",
            "review_score",
            "build_time_estimate",
            "success_score",
            "demand_dissatisfaction",
            "demand_dissatisfaction_percentile",
            "similar_apps",
            "success_reasoning",
        ]
        if risk_metric == "success_per_week":
            display_columns.append("success_per_week")
        display_table = filtered_opps[display_columns].rename(
            columns={
                "name": "App",
                "category_segment": "Category",
                "price": "Price",
                "number_of_ratings": "Rating count",
                "review_score": "Avg rating",
                "build_time_estimate": "Build weeks",
                "success_score": "Stage 2 success",
                "success_reasoning": "Reasoning",
                "demand_dissatisfaction": "Demand dissatisfaction",
                "demand_dissatisfaction_percentile": "Demand %tile",
                "similar_apps": "Similar apps",
                "success_per_week": "Success per week",
            }
        )
        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Demand dissatisfaction": st.column_config.NumberColumn(format="%.0f"),
                "Demand %tile": st.column_config.NumberColumn(format="%.0f"),
                "Reasoning": st.column_config.TextColumn(help="Stage 2 reasoning behind the success estimate."),
            },
        )
        with st.expander("Opportunity snapshots", expanded=False):
            for _, row in filtered_opps.iterrows():
                st.markdown(f"**{row['name']}** — {row['opportunity_note']}")
                neighbors = row.get("neighbors") or []
                if neighbors:
                    neighbor_summary = ", ".join(
                        f"{entry['name']} ({entry['similarity']:.2f})"
                        for entry in neighbors[:5]
                    )
                    st.caption(f"Similar: {neighbor_summary}")
        st.caption(
            "Apps meeting the configured high-price, high-volume, low-rating, low-effort criteria with large dissatisfied audiences. Sort to inspect the derivative angle that matters most."
        )

        top_per_category_all = (
            working.sort_values("demand_dissatisfaction", ascending=False)
            .groupby("category_clean", as_index=False)
            .first()
        )
        if not top_per_category_all.empty:
            top_per_category_all = top_per_category_all.assign(
                opportunity_note=top_per_category_all.apply(
                    lambda row: (
                        f"${row['price']:.2f} {row['category_segment']} app with "
                        f"{int(row['number_of_ratings']):,} ratings averaging {row['review_score']:.1f}/5; "
                        f"{row['build_time_estimate']:.1f} build weeks, success {row['success_score']:.0f}, "
                        f"demand dissatisfaction {row['demand_dissatisfaction']:.0f} ({row['demand_dissatisfaction_percentile']:.0f}th %tile)."
                    ),
                    axis=1,
                )
            )
            st.markdown("### Top opportunities by category")
            st.caption("Skim for the category leaders with the most dissatisfied demand across the full dataset.")
            top_cols = [
                "category_clean",
                "name",
                "price",
                "number_of_ratings",
                "review_score",
                "build_time_estimate",
                "success_score",
                "demand_dissatisfaction",
                "demand_dissatisfaction_percentile",
                "opportunity_note",
            ]
            if risk_metric == "success_per_week":
                top_cols.insert(-1, "success_per_week")
            st.dataframe(
                top_per_category_all[top_cols].rename(
                    columns={
                        "category_clean": "Category",
                        "name": "App",
                        "price": "Price",
                        "number_of_ratings": "Rating count",
                        "review_score": "Avg rating",
                        "build_time_estimate": "Build weeks",
                        "success_score": "Stage 2 success",
                        "demand_dissatisfaction": "Demand dissatisfaction",
                        "demand_dissatisfaction_percentile": "Demand %tile",
                        "success_per_week": "Success per week",
                        "opportunity_note": "Snapshot",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.markdown("---")
    st.subheader("Competitor density overview")
    st.caption(
        "Use this table to understand how densely the current filters populate each niche. High counts signal crowded segments where differentiation matters; low counts hint at underserved spaces."
    )
    density = df.groupby(["category_clean", "price_tier"], as_index=False).agg(
        apps=("track_id", "nunique"),
        total_ratings=("number_of_ratings", "sum"),
        avg_rating=("review_score", "mean"),
    )
    density.sort_values("apps", ascending=False, inplace=True)
    st.dataframe(
        density.rename(
            columns={
                "category_clean": "Category",
                "price_tier": "Tier",
                "apps": "App count",
                "total_ratings": "Total ratings",
                "avg_rating": "Avg rating",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="App Store Opportunity Explorer",
        page_icon="📱",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] {
            width: 26rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("App Store Opportunity Explorer")
    st.caption(
        "Visualise Stage 2 LLM scores to identify high-upside, low-effort app opportunities."
    )

    try:
        df = load_data()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    except Exception as err:  # noqa: BLE001
        st.exception(err)
        st.stop()

    filters = render_sidebar(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No apps match the current filter combination. Adjust your criteria and try again.")
        st.stop()

    render_metrics(filtered_df)

    feature_df = load_feature_table()
    neighbor_lookup = load_neighbors()
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
        st.markdown(
            """
            **Tips**
            - Toggle legend entries to isolate categories.
            - Hover points for developer details, pricing, and ratings volume.
            - Quick wins are shaded in green (≤12 weeks build, ≥70 success).
            """
        )
        st.caption("This scatter compares effort vs. success for the filtered set; use the controls to explore variations.")

    with tab_scatter_3d:
        st.caption(
            "Rotate and zoom to inspect relationships between build effort, success, and a third metric."
        )
        z_metric = st.selectbox(
            "Z-axis metric",
            options=[
                "success_per_week",
                "number_of_ratings",
                "review_score",
                "number_of_downloads",
            ],
            format_func=lambda x: {
                "success_per_week": "Success per week",
                "number_of_ratings": "Rating count",
                "review_score": "Average review",
                "number_of_downloads": "Download proxy",
            }[x],
        )
        color_field = st.selectbox(
            "Colour by",
            options=["category_segment", "price_tier"],
            format_func=lambda x: "Category" if x == "category_segment" else "Price tier",
            index=0,
        )
        fig3d = plot_scatter_3d(filtered_df, z_metric, color_field)
        st.plotly_chart(fig3d, config={"displayModeBar": False})
        st.caption("Use the 3D view to inspect relationships across three metrics; drag to rotate and highlight segments via the legend.")
        st.caption("The 3D view emphasizes multivariate trends; drag to rotate, use the legend to isolate segments.")

    with tab_summary:
        summary_fig = plot_category_summary(filtered_df)
        st.plotly_chart(summary_fig, config={"displayModeBar": False})
        st.caption("Contrast overall app coverage with quick-win volume for each category tier.")
        st.caption("Grouped bars highlight which categories and price tiers dominate within the current filters.")
        st.caption("Use this grouped view to see which categories (and price tiers) dominate the current filters.")

    with tab_distribution:
        dist_fig = plot_success_distribution(filtered_df)
        st.plotly_chart(dist_fig, config={"displayModeBar": False})
        st.caption("Box plot shows distribution of success scores split by free vs paid tiers.")
        st.caption("Helps compare traction spread between free and paid offerings under the current filters.")
        st.caption("Helps compare traction spread between free and paid offerings under the current filters.")

    with tab_table:
        render_top_table(filtered_df)

    clusters_df, members_df = load_cluster_data()
    with tab_clusters:
        render_similarity_clusters(filtered_df, clusters_df, members_df)

    with tab_opportunities:
        render_opportunity_finder(filtered_df, neighbor_lookup)

    if tab_deltas is not None and feature_df is not None:
        with tab_deltas:
            st.subheader("Snapshot deltas (experimental)")
            st.caption(
                "Comparing successive runs per app. Data sourced from src/local/analysis/build_deltas.py."
            )
            st.caption("Track how success and rank metrics shift between runs. Use the metric toggle to focus on score or leaderboard movement.")

            delta_view = feature_df.copy()
            if filters["run_ids"]:
                delta_view = delta_view[delta_view["run_id"].isin(filters["run_ids"])]

            if delta_view.empty:
                st.info("No delta data available for the current run filters.")
            else:
                metric_choice = st.radio(
                    "Metric",
                    options=["Success score", "Rank position"],
                    index=0,
                    horizontal=True,
                )

                metric_config = {
                    "Success score": {
                        "col": "delta_success",
                        "label": "Δ success",
                        "mean_format": lambda v: f"{v:+.2f}" if v is not None else "--",
                        "run_title": "Average Δ success by run",
                        "run_ylabel": "Avg Δ success",
                        "category_title": "Top categories by avg Δ success",
                        "improve_title": "Top positive movers (Δ success)",
                        "decline_title": "Top negative movers (Δ success)",
                        "improve_sort": lambda df: df.nlargest(10, "delta_success"),
                        "decline_sort": lambda df: df.nsmallest(10, "delta_success"),
                        "note": "Positive values indicate rising success score.",
                    },
                    "Rank position": {
                        "col": "delta_rank",
                        "label": "Δ rank (negative improves)",
                        "mean_format": lambda v: f"{v:+.2f}" if v is not None else "--",
                        "run_title": "Average Δ rank by run",
                        "run_ylabel": "Avg Δ rank (negative improves)",
                        "category_title": "Top categories by avg Δ rank",
                        "improve_title": "Top rank improvements",
                        "decline_title": "Top rank declines",
                        "improve_sort": lambda df: df.nsmallest(10, "delta_rank"),
                        "decline_sort": lambda df: df.nlargest(10, "delta_rank"),
                        "note": "Negative values mean the app moved up the leaderboard.",
                    },
                }

                config = metric_config[metric_choice]
                delta_col = config["col"]

                comparables = delta_view[~delta_view["is_new_track"]]

                col_a, col_b, col_c, col_d = st.columns(4)
                delta_series = comparables[delta_col].dropna()
                avg_delta = delta_series.mean() if not delta_series.empty else None
                col_a.metric(
                    config["label"],
                    config["mean_format"](avg_delta),
                    help=config["note"],
                )
                col_b.metric(
                    "Avg Δ build time",
                    f"{comparables['delta_build_time'].dropna().mean():+.2f}" if not comparables["delta_build_time"].dropna().empty else "--",
                )
                col_c.metric(
                    "New apps",
                    int(delta_view["is_new_track"].sum()),
                )
                col_d.metric(
                    "Median days between runs",
                    f"{comparables['days_since_prev'].dropna().median():.1f}" if not comparables["days_since_prev"].dropna().empty else "--",
                )

                run_summary = (
                    comparables.dropna(subset=["prev_run_id"])  # ensure prior exists
                    .groupby(["run_id", "run_created_at"], as_index=False)
                    .agg(
                        avg_delta_metric=(delta_col, "mean"),
                        avg_delta_build=("delta_build_time", "mean"),
                        returning_apps=("track_id", "count"),
                    )
                )
                run_new = (
                    delta_view.groupby(["run_id", "run_created_at"], as_index=False)
                    .agg(new_apps=("is_new_track", "sum"))
                )

                col_line, col_bar = st.columns(2)
                if not run_summary.empty:
                    fig_delta = px.line(
                        run_summary,
                        x="run_created_at",
                        y="avg_delta_metric",
                        markers=True,
                        title=config["run_title"],
                    )
                    fig_delta.update_layout(xaxis_title="Run", yaxis_title=config["run_ylabel"])
                    col_line.plotly_chart(fig_delta, config={"displayModeBar": False})
                else:
                    col_line.info("Not enough data to compute average deltas yet.")

                if not run_new.empty:
                    fig_new = px.bar(
                        run_new,
                        x="run_created_at",
                        y="new_apps",
                        title="New apps per run",
                    )
                    fig_new.update_layout(xaxis_title="Run", yaxis_title="# New apps")
                    col_bar.plotly_chart(fig_new, config={"displayModeBar": False})
                else:
                    col_bar.info("No run data to display new app counts.")

                top_gain = config["improve_sort"](comparables.dropna(subset=[delta_col]))
                top_loss = config["decline_sort"](comparables.dropna(subset=[delta_col]))

                movers_cols = ["run_id", "track_id", "name", "run_created_at"]
                if metric_choice == "Rank position":
                    movers_cols.extend(["best_rank", "prev_rank", "delta_rank"])
                else:
                    movers_cols.extend(["success_score", "prev_success_score", "delta_success"])
                movers_cols.append("days_since_prev")

                col_gain, col_loss = st.columns(2)
                col_gain.markdown(f"**{config['improve_title']}**")
                col_gain.dataframe(top_gain[movers_cols] if not top_gain.empty else pd.DataFrame(columns=movers_cols), width="stretch", hide_index=True)
                col_loss.markdown(f"**{config['decline_title']}**")
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
                        title=config["category_title"],
                    )
                    fig_cat.update_layout(xaxis_title="Category", yaxis_title=config["run_ylabel"])
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
    elif tab_deltas is None:
        st.info(
            "Delta table not available. Run python -m src.local.analysis.build_deltas to populate app_snapshot_deltas."
        )


if __name__ == "__main__":
    main()
