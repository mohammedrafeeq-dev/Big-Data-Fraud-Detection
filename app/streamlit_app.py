# ============================================================
# streamlit_app.py — Main Streamlit Application Entry Point
# ============================================================
"""
Professional Streamlit application for Credit Card Fraud Detection.
Features a modern dark UI with sidebar navigation across 5 pages.

Usage:
    cd app
    streamlit run streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Big Data Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ──────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1C23 0%, #252830 100%);
        border: 1px solid #2A2D35;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #6C63FF;
    }

    /* Glassmorphism container */
    .glass-card {
        background: rgba(26, 28, 35, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }

    /* Prediction result */
    .prediction-legit {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.15), rgba(46, 213, 115, 0.05));
        border: 1px solid rgba(46, 213, 115, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .prediction-fraud {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.15), rgba(255, 71, 87, 0.05));
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #161922 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5A52E0);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF, #6C63FF);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2rem; margin: 0;'>🛡️</h1>
        <h2 style='font-size: 1.2rem; margin: 8px 0; 
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;'>
            Fraud Detection
        </h2>
        <p style='color: #666; font-size: 0.8rem;'>Big Data Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 EDA Dashboard", "🔮 Predict",
         "📈 Model Comparison", "📁 Upload Dataset"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <p style='color: #555; font-size: 0.75rem;'>
            Built with PySpark MLlib<br>
            © 2025 Big Data Project
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ──────────────────────────────────────────────
if page == "🏠 Home":
    from pages.home import render
    render()
elif page == "📊 EDA Dashboard":
    from pages.eda_dashboard import render
    render()
elif page == "🔮 Predict":
    from pages.predict import render
    render()
elif page == "📈 Model Comparison":
    from pages.model_comparison import render
    render()
elif page == "📁 Upload Dataset":
    from pages.upload_dataset import render
    render()
