# ============================================================
# pages/model_comparison.py — Model Comparison Dashboard
# ============================================================
"""Side-by-side model metrics, charts, and best model analysis."""

import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render():
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <h1 style='font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📈 Model Comparison
        </h1>
        <p style='color: #888;'>Compare performance across all trained classification models</p>
    </div>
    """, unsafe_allow_html=True)

    metrics_file = PROJECT_ROOT / "models" / "model_metrics.json"
    viz_dir = PROJECT_ROOT / "outputs" / "visualizations"

    if not metrics_file.exists():
        st.warning("⚠️ No model metrics found. Run the pipeline first:")
        st.code("python scripts/run_pipeline.py", language="bash")
        return

    metrics = json.loads(metrics_file.read_text())
    model_keys = [k for k in metrics if k != "best_model"]
    best = metrics.get("best_model", {})

    # Best model banner
    if best:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(108, 99, 255, 0.15), rgba(0, 212, 170, 0.15));
            border: 1px solid rgba(108, 99, 255, 0.3); border-radius: 16px; padding: 24px;
            text-align: center; margin-bottom: 24px;'>
            <h3 style='color: #6C63FF; margin: 0;'>🏆 Best Model: {best.get("name", "—")}</h3>
            <p style='color: #AAA; margin: 8px 0 0 0;'>
                F1 Score: <strong style='color: #00D4AA;'>{best.get("f1_score", 0):.4f}</strong>
                &nbsp;|&nbsp;
                ROC-AUC: <strong style='color: #00D4AA;'>{best.get("roc_auc", 0):.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics comparison table
    st.markdown("<div class='section-header'>📊 Metrics Summary</div>", unsafe_allow_html=True)

    table_data = []
    for key in model_keys:
        m = metrics[key]
        is_best = key == best.get("key", "")
        table_data.append({
            "Model": ("🏆 " if is_best else "") + m["model_name"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1 Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Interactive bar chart
    st.markdown("<div class='section-header'>📊 Visual Comparison</div>", unsafe_allow_html=True)

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    colors = ["#6C63FF", "#00D4AA", "#FF6B6B", "#FFA502"]

    fig = go.Figure()
    for i, key in enumerate(model_keys):
        m = metrics[key]
        fig.add_trace(go.Bar(
            name=m["model_name"],
            x=metric_labels,
            y=[m.get(mn, 0) for mn in metric_names],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1C23",
        font_color="#FAFAFA",
        title="Model Performance Comparison",
        yaxis_title="Score", yaxis_range=[0, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("<div class='section-header'>🎯 Radar Comparison</div>", unsafe_allow_html=True)

    radar_fig = go.Figure()
    for i, key in enumerate(model_keys):
        m = metrics[key]
        values = [m.get(mn, 0) for mn in metric_names] + [m.get(metric_names[0], 0)]
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=m["model_name"],
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))

    radar_fig.update_layout(
        polar=dict(
            bgcolor="#1A1C23",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        paper_bgcolor="#0E1117", font_color="#FAFAFA",
        showlegend=True, height=500,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # Confusion matrices image
    cm_img = viz_dir / "confusion_matrices.png"
    if cm_img.exists():
        st.markdown("<div class='section-header'>🔢 Confusion Matrices</div>",
                    unsafe_allow_html=True)
        st.image(str(cm_img), use_container_width=True)

    # Saved comparison chart
    comp_img = viz_dir / "model_comparison.png"
    if comp_img.exists():
        with st.expander("📊 Static Comparison Chart"):
            st.image(str(comp_img), use_container_width=True)

    # Per-model details
    st.markdown("<div class='section-header'>🔍 Detailed Model Analysis</div>",
                unsafe_allow_html=True)

    for key in model_keys:
        m = metrics[key]
        cm = m.get("confusion_matrix", {})
        with st.expander(f"{'🏆 ' if key == best.get('key') else ''}{m['model_name']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Classification Metrics**")
                for label, mk in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                  ("Recall", "recall"), ("F1 Score", "f1_score"),
                                  ("ROC-AUC", "roc_auc")]:
                    st.write(f"- {label}: **{m.get(mk, 0):.4f}**")
            with c2:
                st.markdown("**Confusion Matrix**")
                st.write(f"- True Positives: **{cm.get('TP', 0):,}**")
                st.write(f"- True Negatives: **{cm.get('TN', 0):,}**")
                st.write(f"- False Positives: **{cm.get('FP', 0):,}**")
                st.write(f"- False Negatives: **{cm.get('FN', 0):,}**")
