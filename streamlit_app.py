#!/usr/bin/env python3
"""Streamlit dashboard for exploring Stage 2 success vs build time scores."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path("exports") / "app_store_apps.db"

AXIS_OPTIONS = {
    "build_time_estimate": "Estimated build time (weeks)",
    "success_score": "Success score (0-100)",
    "success_per_week": "Success per week",
    "number_of_ratings": "Rating count",
    "review_score": "Average review",
    "price": "Price (USD)",
    "number_of_downloads": "Download proxy (ratings)",
}


@lru_cache(maxsize=1)
def load_data(db_path: Path) -> pd.DataFrame:
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
                chart_memberships,
                scraped_at
            FROM apps
            WHERE build_time_estimate IS NOT NULL
              AND success_score IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
    df["price"] = df["price"].fillna(0.0)
    df["price_tier"] = np.where(df["price"] > 0, "Paid", "Free")
    df["category_clean"] = df["category"].fillna("Unknown")
    df["category_segment"] = df["category_clean"] + " (" + df["price_tier"] + ")"
    df["success_per_week"] = df["success_score"] / df["build_time_estimate"].replace(0, np.nan)
    df["confidence_proxy"] = np.log10(df["number_of_ratings"].fillna(0) + 1)
    df["bubble_size"] = 20 + df["confidence_proxy"] * 25
    df["quick_win"] = (df["build_time_estimate"] <= 12) & (df["success_score"] >= 70)
    return df


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    categories = sorted(df["category_segment"].dropna().unique())
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
        "ratings_range": ratings_range,
        "build_range": build_range,
        "success_range": success_range,
        "quick_win_only": quick_win_only,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df[
        df["category_segment"].isin(filters["categories"])
        & df["price_tier"].isin(filters["price_tiers"])
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
                "developer",
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
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="App Store Opportunity Explorer",
        page_icon="📱",
        layout="wide",
    )
    st.title("App Store Opportunity Explorer")
    st.caption(
        "Visualise Stage 2 LLM scores to identify high-upside, low-effort app opportunities."
    )

    try:
        df = load_data(DB_PATH)
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

    tab_scatter, tab_scatter_3d, tab_summary, tab_distribution, tab_table = st.tabs([
        "Scatter plot",
        "3D scatter",
        "Category summary",
        "Success distribution",
        "Quick-win table",
    ])

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
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Use the legend and lasso/box select tools to inspect specific cohorts."
        )

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
        st.plotly_chart(fig3d, use_container_width=True)

    with tab_summary:
        summary_fig = plot_category_summary(filtered_df)
        st.plotly_chart(summary_fig, use_container_width=True)
        st.caption("Contrast overall app coverage with quick-win volume for each category tier.")

    with tab_distribution:
        dist_fig = plot_success_distribution(filtered_df)
        st.plotly_chart(dist_fig, use_container_width=True)
        st.caption("Box plot shows distribution of success scores split by free vs paid tiers.")

    with tab_table:
        render_top_table(filtered_df)
        st.markdown(
            """
            **Tips**
            - Toggle legend entries on other tabs to isolate categories.
            - Hover points for developer details, pricing, and ratings volume.
            - Quick wins are shaded in green (≤12 weeks build, ≥70 success).
            """
        )


if __name__ == "__main__":
    main()
