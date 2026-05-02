# ============================================================
# pages/home.py — Home / Project Overview Page
# ============================================================
"""Project overview, architecture summary, and key metrics."""

import streamlit as st
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render():
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='font-size: 2.8rem; font-weight: 700; margin: 0;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;'>
            Credit Card Fraud Detection
        </h1>
        <p style='font-size: 1.1rem; color: #888; margin-top: 12px; max-width: 700px; margin-left: auto; margin-right: auto;'>
            An end-to-end Big Data analytics pipeline powered by Apache Spark MLlib — 
            from data acquisition to real-time prediction deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    metrics_file = PROJECT_ROOT / "models" / "model_metrics.json"
    validation_file = PROJECT_ROOT / "outputs" / "validation_report.json"

    total_rows = "500,000+"
    num_features = "30+"
    best_model = "—"
    best_f1 = "—"

    if validation_file.exists():
        val = json.loads(validation_file.read_text())
        total_rows = f"{val.get('total_rows', 500000):,}"
        num_features = str(val.get("total_columns", 31))

    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        bm = metrics.get("best_model", {})
        best_model = bm.get("name", "—")
        f1 = bm.get("f1_score", 0)
        best_f1 = f"{f1:.4f}" if f1 else "—"

    cols = st.columns(4)
    card_data = [
        ("📊", total_rows, "TRANSACTIONS"),
        ("🔢", num_features, "FEATURES"),
        ("🏆", best_model, "BEST MODEL"),
        ("🎯", best_f1, "F1 SCORE"),
    ]
    for col, (icon, value, label) in zip(cols, card_data):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2rem;'>{icon}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>📋 Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <p>This project implements a <strong>production-grade fraud detection system</strong> 
            using Apache Spark's distributed computing framework. The pipeline processes 
            500,000+ credit card transactions to identify fraudulent activity with high accuracy.</p>
            
            <h4 style='color: #6C63FF; margin-top: 16px;'>Key Highlights</h4>
            <ul>
                <li>🔄 End-to-end ML pipeline with PySpark MLlib</li>
                <li>📊 Comprehensive EDA with 7+ visualization types</li>
                <li>🧠 4 classification models trained and compared</li>
                <li>⚡ Scalable preprocessing with Spark DataFrames</li>
                <li>🚀 Real-time prediction via Streamlit interface</li>
                <li>📁 Batch prediction via CSV upload</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>🏗️ Architecture</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #00D4AA;'>Pipeline Stages</h4>
            <ol>
                <li><strong>Data Acquisition</strong> — Kaggle API / Synthetic Generation</li>
                <li><strong>Data Validation</strong> — Schema enforcement & quality checks</li>
                <li><strong>EDA</strong> — Statistical analysis & visualizations</li>
                <li><strong>Feature Engineering</strong> — Log transforms, z-scores, interactions</li>
                <li><strong>Preprocessing</strong> — Cleaning, scaling, vector assembly</li>
                <li><strong>Model Training</strong> — LR, RF, DT, GBT classifiers</li>
                <li><strong>Evaluation</strong> — Metrics comparison & best model selection</li>
                <li><strong>Deployment</strong> — Streamlit web application</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(4)
    techs = [
        ("⚡ Apache Spark", "Distributed data processing and MLlib for scalable machine learning"),
        ("🐍 Python", "Core programming language with pandas, numpy, and scikit-learn"),
        ("📊 Plotly & Seaborn", "Interactive and publication-quality data visualizations"),
        ("🚀 Streamlit", "Modern web framework for ML application deployment"),
    ]
    for col, (title, desc) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='min-height: 140px;'>
                <h4 style='color: #6C63FF; margin: 0 0 8px 0;'>{title}</h4>
                <p style='color: #AAA; font-size: 0.85rem; margin: 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Dataset Info
    st.markdown("<div class='section-header'>📦 Dataset Description</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <table style='width: 100%; color: #CCC;'>
            <tr><td style='padding: 8px; color: #888;'>Source</td>
                <td style='padding: 8px;'>Kaggle — Credit Card Fraud Detection 2023</td></tr>
            <tr><td style='padding: 8px; color: #888;'>Records</td>
                <td style='padding: 8px;'>500,000+ transactions</td></tr>
            <tr><td style='padding: 8px; color: #888;'>Features</td>
                <td style='padding: 8px;'>V1–V28 (PCA), Amount, Class</td></tr>
            <tr><td style='padding: 8px; color: #888;'>Target</td>
                <td style='padding: 8px;'>Binary: 0 = Legitimate, 1 = Fraudulent</td></tr>
            <tr><td style='padding: 8px; color: #888;'>Challenge</td>
                <td style='padding: 8px;'>Highly imbalanced (~1.7% fraud rate)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
