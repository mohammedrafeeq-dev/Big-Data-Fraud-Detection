# ============================================================
# pages/upload_dataset.py — CSV Upload & Batch Prediction
# ============================================================
"""
Allows users to upload their own CSV dataset, automatically
preprocesses it, and runs batch predictions with selected models.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_available_models():
    """Check which models exist on disk."""
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


def _run_batch_prediction(df: pd.DataFrame, model_keys: list) -> dict:
    """Run predictions using selected models on the uploaded data."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.classification import (
            LogisticRegressionModel,
            RandomForestClassificationModel,
            DecisionTreeClassificationModel,
            GBTClassificationModel,
        )
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.sql import functions as F
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.config import DataConfig, Paths

        spark = SparkSession.builder.appName("BatchPrediction").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        # Convert to Spark DataFrame
        sdf = spark.createDataFrame(df)

        # Drop id if exists
        if "id" in sdf.columns:
            sdf = sdf.drop("id")
        if "Class" in sdf.columns:
            sdf = sdf.drop("Class")

        # Feature engineering
        if "Amount" in sdf.columns:
            sdf = sdf.withColumn("log_amount", F.log1p(F.col("Amount")))
            sdf = sdf.withColumn("amount_zscore", F.lit(0.0))
            sdf = sdf.withColumn(
                "amount_bin",
                F.when(F.col("Amount") <= 10, 0.0)
                .when(F.col("Amount") <= 50, 1.0)
                .when(F.col("Amount") <= 200, 2.0)
                .when(F.col("Amount") <= 1000, 3.0)
                .otherwise(4.0)
            )

        if "V1" in sdf.columns and "V3" in sdf.columns:
            sdf = sdf.withColumn("V1_V3_interaction", F.col("V1") * F.col("V3"))
        if "V14" in sdf.columns and "V17" in sdf.columns:
            sdf = sdf.withColumn("V14_V17_interaction", F.col("V14") * F.col("V17"))

        # Assemble features
        feature_cols = [c for c in sdf.columns if c not in ["prediction", "probability", "rawPrediction"]]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        sdf = assembler.transform(sdf)

        # Scale
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(sdf)
        sdf = scaler_model.transform(sdf)

        # Predict with each model
        model_classes = {
            "lr": (LogisticRegressionModel, Paths.LR_MODEL_DIR),
            "rf": (RandomForestClassificationModel, Paths.RF_MODEL_DIR),
            "dt": (DecisionTreeClassificationModel, Paths.DT_MODEL_DIR),
            "gbt": (GBTClassificationModel, Paths.GBT_MODEL_DIR),
        }

        results = {}
        for key in model_keys:
            model_class, model_path = model_classes[key]
            model = model_class.load(str(model_path))
            preds = model.transform(sdf)
            pred_pdf = preds.select("prediction").toPandas()
            results[key] = pred_pdf["prediction"].astype(int).tolist()

        return results

    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return {}


def render():
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <h1 style='font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📁 Upload Dataset
        </h1>
        <p style='color: #888;'>Upload a CSV file for automatic preprocessing and batch prediction</p>
    </div>
    """, unsafe_allow_html=True)

    available_models = _get_available_models()
    if not available_models:
        st.warning("⚠️ No trained models found. Run the pipeline first.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV with the same features as the training data (V1-V28, Amount)",
    )

    if uploaded_file is not None:
        # Load and preview
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicates", int(df.duplicated().sum()))

        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        # Column validation
        expected = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        found = [c for c in expected if c in df.columns]
        missing = [c for c in expected if c not in df.columns]

        if missing:
            st.warning(f"⚠️ Missing expected columns: {', '.join(missing[:10])}")
            st.info("The model will use available columns for prediction.")

        st.divider()

        # Model selection
        st.markdown("<div class='section-header'>🧠 Select Models</div>", unsafe_allow_html=True)
        selected = st.multiselect(
            "Choose models for prediction",
            options=list(available_models.keys()),
            default=list(available_models.keys()),
            format_func=lambda k: available_models[k],
        )

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            if not selected:
                st.warning("Please select at least one model.")
                return

            with st.spinner("Running batch predictions..."):
                results = _run_batch_prediction(df, selected)

            if results:
                st.markdown("<div class='section-header'>📊 Prediction Results</div>",
                            unsafe_allow_html=True)

                result_df = df.copy()
                for key, preds in results.items():
                    col_name = f"Pred_{available_models[key]}"
                    result_df[col_name] = preds[:len(result_df)]

                st.dataframe(result_df.head(50), use_container_width=True, hide_index=True)

                # Summary statistics
                st.markdown("<div class='section-header'>📈 Prediction Summary</div>",
                            unsafe_allow_html=True)
                summary_cols = st.columns(len(results))
                for col, (key, preds) in zip(summary_cols, results.items()):
                    total = len(preds)
                    fraud_count = sum(preds)
                    legit_count = total - fraud_count
                    with col:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align: center;'>
                            <h4 style='color: #6C63FF;'>{available_models[key]}</h4>
                            <p style='color: #2ED573; font-size: 1.2rem;'>
                                ✅ {legit_count:,} Legitimate
                            </p>
                            <p style='color: #FF4757; font-size: 1.2rem;'>
                                🚨 {fraud_count:,} Fraudulent
                            </p>
                            <p style='color: #888;'>
                                Fraud Rate: {fraud_count/total*100:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Download results
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        # Instructions
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #6C63FF;'>📋 Expected CSV Format</h4>
            <p style='color: #AAA;'>Your CSV should contain the following columns:</p>
            <ul style='color: #CCC;'>
                <li><strong>V1 – V28</strong>: PCA-transformed numerical features</li>
                <li><strong>Amount</strong>: Transaction amount in dollars</li>
                <li><strong>Class</strong> (optional): Actual labels for comparison</li>
            </ul>
            <p style='color: #888; margin-top: 12px;'>
                The system will automatically preprocess the data, handle missing values,
                engineer features, and generate predictions using your selected models.
            </p>
        </div>
        """, unsafe_allow_html=True)
