# ============================================================
# run_pipeline.py — Master Pipeline Orchestration
# ============================================================
"""
Orchestrates the complete end-to-end fraud detection pipeline:
1. Data acquisition & loading
2. Exploratory Data Analysis
3. Feature Engineering
4. Preprocessing
5. Model Training (4 models)
6. Model Evaluation & Comparison
7. Artifact saving

Usage:
    python scripts/run_pipeline.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths
from src.utils import logger, print_header, save_json
from src.spark_session import create_spark_session, stop_spark_session
from src.data_ingestion import load_raw_data, validate_data
from src.eda import compute_statistics, generate_all_visualizations, generate_business_insights
from src.feature_engineering import run_feature_engineering
from src.preprocessing import run_full_preprocessing
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models, generate_evaluation_charts


def main():
    """Execute the complete pipeline."""
    pipeline_start = time.time()

    print_header("BIG DATA FRAUD DETECTION PIPELINE")
    logger.info("Initializing pipeline...")

    # Create all directories
    Paths.create_all()

    # ── Step 1: Initialize Spark ─────────────────────────────
    print_header("STEP 1: Spark Session")
    spark = create_spark_session()
    logger.info(f"Spark version: {spark.version}")

    try:
        # ── Step 2: Data Ingestion ───────────────────────────
        print_header("STEP 2: Data Ingestion")
        df = load_raw_data(spark)

        # ── Step 3: Data Validation ──────────────────────────
        print_header("STEP 3: Data Validation")
        validation = validate_data(df)
        save_json(validation, Paths.OUTPUTS_DIR / "validation_report.json")

        # ── Step 4: EDA ──────────────────────────────────────
        print_header("STEP 4: Exploratory Data Analysis")
        stats = compute_statistics(df)
        generate_all_visualizations(df)
        insights = generate_business_insights(df, validation)

        eda_report = {"statistics": stats, "insights": insights}
        save_json(eda_report, Paths.OUTPUTS_DIR / "eda_report.json")
        logger.info("Business Insights:")
        for i, insight in enumerate(insights, 1):
            logger.info(f"  {i}. {insight}")

        # ── Step 5: Feature Engineering ──────────────────────
        print_header("STEP 5: Feature Engineering")
        df = run_feature_engineering(df)

        # ── Step 6: Preprocessing ────────────────────────────
        print_header("STEP 6: Preprocessing")
        train_df, test_df = run_full_preprocessing(df)

        # ── Step 7: Model Training ───────────────────────────
        print_header("STEP 7: Model Training")
        models = train_all_models(train_df)

        # ── Step 8: Model Evaluation ─────────────────────────
        print_header("STEP 8: Model Evaluation")
        all_metrics = evaluate_all_models(models, test_df)
        generate_evaluation_charts(all_metrics)

        # ── Pipeline Summary ─────────────────────────────────
        elapsed = time.time() - pipeline_start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        print_header("PIPELINE COMPLETE")
        logger.info(f"Total execution time: {minutes}m {seconds:.1f}s")
        logger.info(f"Models trained: {len(models)}")

        if "best_model" in all_metrics:
            best = all_metrics["best_model"]
            logger.info(f"Best model: {best['name']} "
                       f"(F1={best['f1_score']:.4f}, AUC={best['roc_auc']:.4f})")

        logger.info(f"\nOutputs saved to: {Paths.OUTPUTS_DIR}")
        logger.info(f"Models saved to:  {Paths.MODELS_DIR}")
        logger.info(f"\nTo launch the Streamlit app:")
        logger.info(f"  streamlit run app/streamlit_app.py")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
