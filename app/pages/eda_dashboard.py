# ============================================================
# pages/eda_dashboard.py — EDA Dashboard Page
# ============================================================
"""Interactive EDA visualizations and statistical summaries."""

import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render():
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <h1 style='font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📊 Exploratory Data Analysis
        </h1>
        <p style='color: #888;'>Comprehensive analysis of the credit card transactions dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Load saved visualizations
    viz_dir = PROJECT_ROOT / "outputs" / "visualizations"
    eda_file = PROJECT_ROOT / "outputs" / "eda_report.json"
    val_file = PROJECT_ROOT / "outputs" / "validation_report.json"

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "🔥 Correlations", "📊 Class Analysis", "💡 Insights"
    ])

    with tab1:
        st.markdown("<div class='section-header'>Transaction Amount Distribution</div>",
                    unsafe_allow_html=True)
        img_path = viz_dir / "amount_distribution.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            _generate_plotly_amount_dist()

        st.markdown("<div class='section-header'>Feature Distributions by Class</div>",
                    unsafe_allow_html=True)
        img_path = viz_dir / "feature_distributions.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Run the pipeline first to generate visualizations: `python scripts/run_pipeline.py`")

    with tab2:
        st.markdown("<div class='section-header'>Correlation Heatmap</div>",
                    unsafe_allow_html=True)
        img_path = viz_dir / "correlation_heatmap.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Run the pipeline to generate correlation analysis.")

        st.markdown("<div class='section-header'>Scatter Plots: Amount vs Key Features</div>",
                    unsafe_allow_html=True)
        img_path = viz_dir / "scatter_plots.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Class Distribution</div>",
                        unsafe_allow_html=True)
            img_path = viz_dir / "class_distribution.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                _generate_plotly_class_dist(val_file)

        with col2:
            st.markdown("<div class='section-header'>Amount by Class</div>",
                        unsafe_allow_html=True)
            img_path = viz_dir / "amount_by_class.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)

        st.markdown("<div class='section-header'>Feature Boxplots by Class</div>",
                    unsafe_allow_html=True)
        img_path = viz_dir / "feature_boxplots.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

    with tab4:
        st.markdown("<div class='section-header'>💡 Key Business Insights</div>",
                    unsafe_allow_html=True)

        if eda_file.exists():
            eda = json.loads(eda_file.read_text())
            insights = eda.get("insights", [])
            for i, insight in enumerate(insights, 1):
                st.markdown(f"""
                <div class='glass-card' style='margin: 8px 0;'>
                    <span style='color: #6C63FF; font-weight: 700;'>Insight {i}:</span>
                    <span style='color: #CCC;'> {insight}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass-card'>
                <p style='color: #888;'>Run the pipeline to generate insights:</p>
                <code>python scripts/run_pipeline.py</code>
            </div>
            """, unsafe_allow_html=True)

        # Validation summary
        if val_file.exists():
            val = json.loads(val_file.read_text())
            st.markdown("<div class='section-header'>📋 Data Quality Report</div>",
                        unsafe_allow_html=True)
            cols = st.columns(4)
            items = [
                ("Total Rows", f"{val.get('total_rows', 0):,}"),
                ("Total Columns", val.get("total_columns", 0)),
                ("Missing Values", val.get("total_nulls", 0)),
                ("Duplicates", f"{val.get('duplicates', 0):,}"),
            ]
            for col, (label, value) in zip(cols, items):
                with col:
                    st.metric(label, value)


def _generate_plotly_class_dist(val_file):
    """Fallback: generate class distribution with Plotly."""
    if val_file.exists():
        val = json.loads(val_file.read_text())
        cd = val.get("class_distribution", {})
        if cd:
            labels = list(cd.keys())
            values = [cd[l]["count"] for l in labels]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                marker_colors=["#2ED573", "#FF4757"],
                hole=0.4
            )])
            fig.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#1A1C23",
                font_color="#FAFAFA", showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            return
    st.info("No class distribution data available.")


def _generate_plotly_amount_dist():
    """Fallback placeholder for amount distribution."""
    st.info("Run the pipeline to generate amount distribution charts.")
