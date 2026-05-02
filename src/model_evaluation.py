# ============================================================
# model_evaluation.py — Model Evaluation & Comparison
# ============================================================
"""
Evaluates trained models using classification metrics:
Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrix.
Generates comparison tables and visualization charts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

try:
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer, save_json
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer, save_json

# Plot style
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
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0E1117",
})

LABEL_COL = DataConfig.LABEL_COL
PRED_COL = DataConfig.PREDICTION_COL
PROB_COL = DataConfig.PROBABILITY_COL


def evaluate_single_model(predictions: DataFrame, model_name: str) -> dict:
    """
    Compute all classification metrics for a single model.

    Returns dict with accuracy, precision, recall, f1, roc_auc,
    and confusion matrix values.
    """
    logger.info(f"  Evaluating: {model_name}")

    # Binary classification evaluator (ROC-AUC)
    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction"
    )
    try:
        roc_auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    except Exception:
        roc_auc = 0.0

    # Multi-class evaluator for standard metrics
    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol=PRED_COL
    )
    accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})

    # Confusion matrix
    cm_df = (
        predictions
        .groupBy(LABEL_COL, PRED_COL)
        .count()
        .orderBy(LABEL_COL, PRED_COL)
        .collect()
    )
    tp = fp = tn = fn = 0
    for row in cm_df:
        actual, predicted, count = row[LABEL_COL], row[PRED_COL], row["count"]
        if actual == 1.0 and predicted == 1.0:
            tp = count
        elif actual == 0.0 and predicted == 1.0:
            fp = count
        elif actual == 0.0 and predicted == 0.0:
            tn = count
        elif actual == 1.0 and predicted == 0.0:
            fn = count

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }

    logger.info(f"    Accuracy={accuracy:.4f} | Precision={precision:.4f} | "
                f"Recall={recall:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")
    return metrics


@timer
def evaluate_all_models(models: dict, test_df: DataFrame) -> dict:
    """
    Evaluate all trained models on the test set.

    Parameters
    ----------
    models : dict
        {key: trained_model} pairs.
    test_df : DataFrame
        Test DataFrame with features and labels.

    Returns
    -------
    dict
        All model metrics, plus a 'best_model' key.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    test_df.cache()
    all_metrics = {}

    for key, model in models.items():
        name = ModelConfig.MODEL_NAMES.get(key, key)
        try:
            predictions = model.transform(test_df)
            metrics = evaluate_single_model(predictions, name)
            all_metrics[key] = metrics
        except Exception as e:
            logger.error(f"  Evaluation failed for {name}: {e}")

    # Determine best model by F1 score
    if all_metrics:
        best_key = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
        best = all_metrics[best_key]
        all_metrics["best_model"] = {
            "key": best_key,
            "name": best["model_name"],
            "f1_score": best["f1_score"],
            "roc_auc": best["roc_auc"],
        }
        logger.info(f"\n  🏆 Best Model: {best['model_name']} "
                     f"(F1={best['f1_score']:.4f}, AUC={best['roc_auc']:.4f})")

    # Save metrics
    save_json(all_metrics, Paths.METRICS_FILE)

    test_df.unpersist()
    return all_metrics


@timer
def generate_evaluation_charts(all_metrics: dict, output_dir=None):
    """Generate comparison visualizations for all models."""
    output_dir = Path(output_dir or Paths.VIZ_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out the 'best_model' key
    model_keys = [k for k in all_metrics if k != "best_model"]
    if not model_keys:
        return

    names = [all_metrics[k]["model_name"] for k in model_keys]
    colors = ["#6C63FF", "#00D4AA", "#FF6B6B", "#FFA502"]

    # ── Metrics Comparison Bar Chart ─────────────────────────
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_labels))
    width = 0.18

    for i, key in enumerate(model_keys):
        values = [all_metrics[key].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, values, width, label=names[i],
               color=colors[i % len(colors)], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_dir / "model_comparison.png"))
    plt.close(fig)

    # ── Confusion Matrix Heatmaps ────────────────────────────
    fig, axes = plt.subplots(1, len(model_keys), figsize=(5 * len(model_keys), 4))
    if len(model_keys) == 1:
        axes = [axes]

    for idx, key in enumerate(model_keys):
        cm = all_metrics[key]["confusion_matrix"]
        matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(names[idx], fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrices.png"))
    plt.close(fig)

    logger.info(f"  Evaluation charts saved to: {output_dir}")
