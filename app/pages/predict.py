# ============================================================
# pages/predict.py — Real-Time Prediction Page
# ============================================================
"""
Allows users to input transaction features and get instant
fraud predictions with confidence scores from trained models.
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_metrics():
    """Load model metrics for display."""
    mf = PROJECT_ROOT / "models" / "model_metrics.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {}


def _get_available_models():
    """Check which models are saved on disk."""
    models_dir = PROJECT_ROOT / "models"
    available = {}
    model_map = {
        "lr": ("logistic_regression", "Logistic Regression"),
        "rf": ("random_forest", "Random Forest"),
        "dt": ("decision_tree", "Decision Tree"),
        "gbt": ("gbt", "Gradient Boosted Trees"),
    }
    for key, (folder, name) in model_map.items():
        path = models_dir / folder
        if path.exists() and any(path.iterdir()):
            available[key] = name
    return available


def _predict_with_spark(features: dict, model_key: str):
    """
    Load a saved model and make a prediction.
    Returns (prediction, probability).
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.classification import (
            LogisticRegressionModel,
            RandomForestClassificationModel,
            DecisionTreeClassificationModel,
            GBTClassificationModel,
        )
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.sql.types import StructType, StructField, DoubleType
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.config import DataConfig, Paths

        # Get or create Spark session
        spark = SparkSession.builder.appName("FraudPrediction").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        # Build a single-row DataFrame
        schema = StructType([StructField(k, DoubleType(), True) for k in features.keys()])
        row_data = [tuple(float(v) for v in features.values())]
        input_df = spark.createDataFrame(row_data, schema)

        # Feature engineering: add derived features
        from pyspark.sql import functions as F
        input_df = input_df.withColumn("log_amount", F.log1p(F.col("Amount")))
        input_df = input_df.withColumn("amount_zscore", F.lit(0.0))  # approximate
        if "V1" in features and "V3" in features:
            input_df = input_df.withColumn("V1_V3_interaction", F.col("V1") * F.col("V3"))
        if "V14" in features and "V17" in features:
            input_df = input_df.withColumn("V14_V17_interaction", F.col("V14") * F.col("V17"))
        input_df = input_df.withColumn(
            "amount_bin",
            F.when(F.col("Amount") <= 10, 0.0)
            .when(F.col("Amount") <= 50, 1.0)
            .when(F.col("Amount") <= 200, 2.0)
            .when(F.col("Amount") <= 1000, 3.0)
            .otherwise(4.0)
        )

        # Assemble features
        all_features = list(features.keys()) + [
            "log_amount", "amount_zscore",
            "V1_V3_interaction", "V14_V17_interaction", "amount_bin"
        ]
        available_cols = [c for c in all_features if c in input_df.columns]
        assembler = VectorAssembler(inputCols=available_cols, outputCol="features", handleInvalid="skip")
        input_df = assembler.transform(input_df)

        # Scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(input_df)
        input_df = scaler_model.transform(input_df)

        # Load model
        model_paths = {
            "lr": (LogisticRegressionModel, Paths.LR_MODEL_DIR),
            "rf": (RandomForestClassificationModel, Paths.RF_MODEL_DIR),
            "dt": (DecisionTreeClassificationModel, Paths.DT_MODEL_DIR),
            "gbt": (GBTClassificationModel, Paths.GBT_MODEL_DIR),
        }
        model_class, model_path = model_paths[model_key]
        model = model_class.load(str(model_path))

        # Predict
        result = model.transform(input_df)
        row = result.select("prediction", "rawPrediction").collect()[0]
        prediction = int(row["prediction"])

        # Extract probability if available
        try:
            prob_row = result.select("probability").collect()[0]
            prob = float(prob_row["probability"][1])  # probability of class 1
        except Exception:
            prob = 1.0 if prediction == 1 else 0.0

        return prediction, prob

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


def render():
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <h1 style='font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🔮 Fraud Prediction
        </h1>
        <p style='color: #888;'>Input transaction details to get an instant fraud assessment</p>
    </div>
    """, unsafe_allow_html=True)

    available_models = _get_available_models()

    if not available_models:
        st.warning("⚠️ No trained models found. Run the pipeline first:")
        st.code("python scripts/run_pipeline.py", language="bash")
        return

    # Model selection
    model_key = st.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        format_func=lambda k: available_models[k],
    )

    # Show model metrics
    metrics = _load_metrics()
    if model_key in metrics:
        m = metrics[model_key]
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("Accuracy", "accuracy"), ("Precision", "precision"),
            ("Recall", "recall"), ("F1", "f1_score"), ("AUC", "roc_auc")
        ]):
            col.metric(label, f"{m.get(key, 0):.4f}")

    st.divider()

    # Feature input form
    st.markdown("<div class='section-header'>Enter Transaction Features</div>",
                unsafe_allow_html=True)

    features = {}
    cols_per_row = 4

    # PCA features in grid
    pca_features = [f"V{i}" for i in range(1, 29)]
    for row_start in range(0, len(pca_features), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(pca_features):
                feat = pca_features[idx]
                features[feat] = col.number_input(
                    feat, value=0.0, step=0.1, format="%.4f", key=f"pred_{feat}"
                )

    # Amount
    st.markdown("---")
    features["Amount"] = st.number_input(
        "💰 Transaction Amount ($)", value=100.0, min_value=0.0,
        max_value=30000.0, step=10.0, format="%.2f"
    )

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Transaction", use_container_width=True):
        with st.spinner("Running prediction..."):
            prediction, probability = _predict_with_spark(features, model_key)

        if prediction is not None:
            if prediction == 0:
                st.markdown(f"""
                <div class='prediction-legit'>
                    <h2 style='color: #2ED573; margin: 0;'>✅ LEGITIMATE</h2>
                    <p style='color: #AAA; font-size: 1.1rem; margin-top: 12px;'>
                        This transaction appears to be legitimate.
                    </p>
                    <p style='color: #2ED573; font-size: 1.4rem; font-weight: 700;'>
                        Confidence: {(1 - probability) * 100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-fraud'>
                    <h2 style='color: #FF4757; margin: 0;'>🚨 FRAUDULENT</h2>
                    <p style='color: #AAA; font-size: 1.1rem; margin-top: 12px;'>
                        This transaction has been flagged as potentially fraudulent.
                    </p>
                    <p style='color: #FF4757; font-size: 1.4rem; font-weight: 700;'>
                        Fraud Probability: {probability * 100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Quick test buttons
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🧪 Quick Test — Use Sample Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Legitimate Sample**")
            st.json({"V1": 1.2, "V2": 0.5, "V14": 0.3, "Amount": 49.99})
        with col2:
            st.markdown("**Fraud Sample**")
            st.json({"V1": -2.5, "V3": -3.1, "V14": -5.2, "V17": -4.1, "Amount": 899.50})
