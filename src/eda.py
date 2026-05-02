# ============================================================
# eda.py — Exploratory Data Analysis
# ============================================================
"""
Comprehensive EDA functions for the fraud detection dataset.
Generates statistics, insights, and publication-quality visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

try:
    from src.config import Paths, DataConfig
    from src.utils import logger, timer, print_header
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Paths, DataConfig
    from src.utils import logger, timer, print_header

# Style configuration
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1A1C23",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#BBBBBB",
    "ytick.color": "#BBBBBB",
    "axes.edgecolor": "#333333",
    "grid.color": "#2A2D35",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0E1117",
})


@timer
def compute_statistics(df: DataFrame) -> dict:
    """Compute comprehensive statistics for all numeric columns."""
    logger.info("Computing dataset statistics...")
    pdf = df.describe().toPandas()
    stats = {}
    for col in pdf.columns[1:]:
        stats[col] = {row["summary"]: row[col] for _, row in pdf.iterrows()}
    return stats


@timer
def generate_all_visualizations(df: DataFrame, output_dir=None):
    """Generate all EDA visualizations and save as PNG."""
    output_dir = Path(output_dir or Paths.VIZ_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to Pandas for plotting (sample if needed)
    total = df.count()
    if total > 100000:
        pdf = df.sample(fraction=100000/total, seed=42).toPandas()
    else:
        pdf = df.toPandas()

    logger.info(f"Generating visualizations from {len(pdf):,} samples...")

    _plot_class_distribution(pdf, output_dir)
    _plot_amount_distribution(pdf, output_dir)
    _plot_correlation_heatmap(pdf, output_dir)
    _plot_feature_boxplots(pdf, output_dir)
    _plot_scatter_plots(pdf, output_dir)
    _plot_amount_by_class(pdf, output_dir)
    _plot_feature_distributions(pdf, output_dir)

    logger.info(f"All visualizations saved to: {output_dir}")


def _plot_class_distribution(pdf, output_dir):
    """Class distribution bar + pie chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = pdf["Class"].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["#2ED573", "#FF4757"]

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Transaction Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + len(pdf) * 0.005, f"{v:,}", ha="center", fontsize=11,
                     color="#FAFAFA", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.2f%%",
                startangle=90, textprops={"color": "#FAFAFA", "fontsize": 11},
                wedgeprops={"edgecolor": "#0E1117", "linewidth": 2})
    axes[1].set_title("Fraud vs Legitimate Ratio", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(output_dir / "class_distribution.png"))
    plt.close(fig)
    logger.info("  ✓ Class distribution chart saved")


def _plot_amount_distribution(pdf, output_dir):
    """Transaction amount distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pdf["Amount"], bins=80, color="#6C63FF", edgecolor="white",
                 linewidth=0.3, alpha=0.85)
    axes[0].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")

    # Log-scale
    axes[1].hist(pdf[pdf["Amount"] > 0]["Amount"], bins=80, color="#00D4AA",
                 edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_title("Amount Distribution (Log Scale)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Frequency (log)")

    plt.tight_layout()
    fig.savefig(str(output_dir / "amount_distribution.png"))
    plt.close(fig)
    logger.info("  ✓ Amount distribution chart saved")


def _plot_correlation_heatmap(pdf, output_dir):
    """Correlation heatmap for top correlated features with target."""
    numeric_cols = [c for c in pdf.columns if c not in ["id"]]
    corr = pdf[numeric_cols].corr()

    # Top 15 features most correlated with Class
    target_corr = corr["Class"].abs().sort_values(ascending=False)
    top_features = target_corr.head(16).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pdf[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Top Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(output_dir / "correlation_heatmap.png"))
    plt.close(fig)
    logger.info("  ✓ Correlation heatmap saved")


def _plot_feature_boxplots(pdf, output_dir):
    """Boxplots for key PCA features by class."""
    key_features = ["V1", "V3", "V4", "V7", "V10", "V12", "V14", "V17"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        sns.boxplot(x="Class", y=feat, data=pdf, ax=axes[idx],
                    palette=["#2ED573", "#FF4757"], width=0.5, fliersize=1)
        axes[idx].set_title(feat, fontsize=12, fontweight="bold")
        axes[idx].set_xticklabels(["Legit", "Fraud"])

    fig.suptitle("Key Feature Distributions by Class", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(output_dir / "feature_boxplots.png"))
    plt.close(fig)
    logger.info("  ✓ Feature boxplots saved")


def _plot_scatter_plots(pdf, output_dir):
    """Scatter plots of Amount vs top PCA features."""
    features = ["V1", "V3", "V14", "V17"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    sample = pdf.sample(min(5000, len(pdf)), random_state=42)
    for idx, feat in enumerate(features):
        legit = sample[sample["Class"] == 0]
        fraud = sample[sample["Class"] == 1]
        axes[idx].scatter(legit[feat], legit["Amount"], c="#2ED573", alpha=0.3,
                          s=5, label="Legit")
        axes[idx].scatter(fraud[feat], fraud["Amount"], c="#FF4757", alpha=0.7,
                          s=15, label="Fraud")
        axes[idx].set_xlabel(feat)
        axes[idx].set_ylabel("Amount")
        axes[idx].set_title(f"Amount vs {feat}", fontsize=12, fontweight="bold")
        axes[idx].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(output_dir / "scatter_plots.png"))
    plt.close(fig)
    logger.info("  ✓ Scatter plots saved")


def _plot_amount_by_class(pdf, output_dir):
    """Amount distribution comparison for fraud vs legitimate."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pdf[pdf["Class"] == 0]["Amount"], bins=60, alpha=0.7, color="#2ED573",
            label="Legitimate", density=True)
    ax.hist(pdf[pdf["Class"] == 1]["Amount"], bins=60, alpha=0.7, color="#FF4757",
            label="Fraudulent", density=True)
    ax.set_title("Amount Distribution: Fraud vs Legitimate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(output_dir / "amount_by_class.png"))
    plt.close(fig)
    logger.info("  ✓ Amount by class chart saved")


def _plot_feature_distributions(pdf, output_dir):
    """Distribution histograms for top 8 features."""
    features = ["V1", "V2", "V3", "V4", "V10", "V12", "V14", "V17"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        axes[idx].hist(pdf[pdf["Class"] == 0][feat], bins=50, alpha=0.6,
                       color="#2ED573", label="Legit", density=True)
        axes[idx].hist(pdf[pdf["Class"] == 1][feat], bins=50, alpha=0.6,
                       color="#FF4757", label="Fraud", density=True)
        axes[idx].set_title(feat, fontsize=11, fontweight="bold")
        axes[idx].legend(fontsize=7)

    fig.suptitle("Feature Distributions by Class", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(output_dir / "feature_distributions.png"))
    plt.close(fig)
    logger.info("  ✓ Feature distributions saved")


@timer
def generate_business_insights(df: DataFrame, validation: dict) -> list:
    """Extract key business insights from the data."""
    insights = []
    cd = validation.get("class_distribution", {})

    if "Fraudulent" in cd:
        fraud_pct = cd["Fraudulent"]["percentage"]
        insights.append(f"Dataset is highly imbalanced: only {fraud_pct}% of transactions are fraudulent.")
        insights.append("Class imbalance handling is critical for model performance.")

    # Amount analysis
    amount_stats = df.select(
        F.mean("Amount").alias("mean"),
        F.stddev("Amount").alias("std"),
        F.max("Amount").alias("max"),
    ).collect()[0]

    fraud_amount = df.filter(F.col("Class") == 1).select(F.mean("Amount")).collect()[0][0]
    legit_amount = df.filter(F.col("Class") == 0).select(F.mean("Amount")).collect()[0][0]

    if fraud_amount and legit_amount:
        insights.append(f"Average fraud amount: ${fraud_amount:,.2f} vs legitimate: ${legit_amount:,.2f}")
        if fraud_amount > legit_amount:
            insights.append("Fraudulent transactions tend to have higher amounts on average.")

    insights.append(f"Maximum transaction amount: ${amount_stats['max']:,.2f}")
    insights.append("PCA-transformed features protect cardholder privacy while enabling ML.")

    return insights
