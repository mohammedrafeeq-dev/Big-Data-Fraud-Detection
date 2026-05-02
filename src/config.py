# ============================================================
# config.py — Central Configuration
# ============================================================
"""
Centralized configuration for the Big Data Fraud Detection pipeline.
All paths, hyperparameters, feature lists, and settings are defined here
to ensure consistency across all modules.
"""

import os
from pathlib import Path


# ────────────────────────────────────────────────────────────
# Project Root Detection
# ────────────────────────────────────────────────────────────
# Dynamically resolve project root regardless of where the script is run from
_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CURRENT_FILE.parent.parent  # src/ → project root


# ────────────────────────────────────────────────────────────
# Directory Paths
# ────────────────────────────────────────────────────────────
class Paths:
    """All filesystem paths used across the project."""

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # Model storage
    MODELS_DIR = PROJECT_ROOT / "models"
    LR_MODEL_DIR = MODELS_DIR / "logistic_regression"
    RF_MODEL_DIR = MODELS_DIR / "random_forest"
    DT_MODEL_DIR = MODELS_DIR / "decision_tree"
    GBT_MODEL_DIR = MODELS_DIR / "gbt"
    METRICS_FILE = MODELS_DIR / "model_metrics.json"

    # Outputs
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    VIZ_DIR = OUTPUTS_DIR / "visualizations"

    # Raw data file
    RAW_CSV = RAW_DATA_DIR / "creditcard_2023.csv"

    # Processed data
    PROCESSED_PARQUET = PROCESSED_DATA_DIR / "creditcard_processed.parquet"
    TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
    TEST_PARQUET = PROCESSED_DATA_DIR / "test.parquet"

    @classmethod
    def create_all(cls):
        """Create all required directories if they don't exist."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and "DIR" in attr_name:
                attr.mkdir(parents=True, exist_ok=True)
        # Also create the viz directory
        cls.VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────
# Dataset Configuration
# ────────────────────────────────────────────────────────────
class DataConfig:
    """Dataset schema and feature definitions."""

    # Kaggle dataset identifier
    KAGGLE_DATASET = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"

    # Target column
    TARGET_COL = "Class"

    # ID column (to be dropped)
    ID_COL = "id"

    # PCA feature columns
    PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1 through V28

    # Amount column
    AMOUNT_COL = "Amount"

    # All input features (after dropping id)
    INPUT_FEATURES = PCA_FEATURES + [AMOUNT_COL]

    # Engineered features (added during feature engineering)
    ENGINEERED_FEATURES = ["log_amount", "amount_zscore"]

    # Final features for model training
    ALL_FEATURES = PCA_FEATURES + [AMOUNT_COL] + ENGINEERED_FEATURES

    # Feature vector column name (for MLlib)
    FEATURES_COL = "features"
    SCALED_FEATURES_COL = "scaled_features"

    # Label column (indexed target)
    LABEL_COL = "label"

    # Prediction column
    PREDICTION_COL = "prediction"
    PROBABILITY_COL = "probability"
    RAW_PREDICTION_COL = "rawPrediction"

    # Synthetic data generation
    SYNTHETIC_NUM_ROWS = 500_000
    SYNTHETIC_FRAUD_RATIO = 0.017  # ~1.7% fraud rate


# ────────────────────────────────────────────────────────────
# Spark Configuration
# ────────────────────────────────────────────────────────────
class SparkConfig:
    """Apache Spark session configuration."""

    APP_NAME = "BigData_FraudDetection"
    MASTER = "local[*]"  # Use all available cores

    # Memory settings
    DRIVER_MEMORY = "4g"
    EXECUTOR_MEMORY = "4g"

    # Performance tuning
    SHUFFLE_PARTITIONS = "8"
    ADAPTIVE_ENABLED = "true"
    SERIALIZER = "org.apache.spark.serializer.KryoSerializer"

    # SQL settings
    LEGACY_TIME_PARSER = "LEGACY"


# ────────────────────────────────────────────────────────────
# Model Hyperparameters
# ────────────────────────────────────────────────────────────
class ModelConfig:
    """Hyperparameters for all MLlib models."""

    # Train/Test split
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    RANDOM_SEED = 42

    # Logistic Regression
    LR_MAX_ITER = 100
    LR_REG_PARAM = 0.01
    LR_ELASTIC_NET = 0.8

    # Random Forest
    RF_NUM_TREES = 100
    RF_MAX_DEPTH = 12
    RF_MAX_BINS = 64
    RF_SUBSAMPLING_RATE = 0.8

    # Decision Tree
    DT_MAX_DEPTH = 10
    DT_MAX_BINS = 64

    # Gradient Boosted Trees
    GBT_MAX_ITER = 50
    GBT_MAX_DEPTH = 8
    GBT_STEP_SIZE = 0.1
    GBT_SUBSAMPLING_RATE = 0.8

    # Model names (for display and storage)
    MODEL_NAMES = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "dt": "Decision Tree",
        "gbt": "Gradient Boosted Trees",
    }


# ────────────────────────────────────────────────────────────
# Streamlit App Configuration
# ────────────────────────────────────────────────────────────
class AppConfig:
    """Streamlit application settings."""

    PAGE_TITLE = "Fraud Detection — Big Data Analytics"
    PAGE_ICON = "🛡️"
    LAYOUT = "wide"

    # Color scheme
    PRIMARY_COLOR = "#6C63FF"
    SECONDARY_COLOR = "#00D4AA"
    DANGER_COLOR = "#FF4757"
    SUCCESS_COLOR = "#2ED573"
    BACKGROUND_DARK = "#0E1117"
    CARD_BG = "#1A1C23"
    TEXT_COLOR = "#FAFAFA"

    # Feature display names for the prediction form
    FEATURE_DESCRIPTIONS = {
        "V1": "V1 (PCA Component 1)",
        "V2": "V2 (PCA Component 2)",
        "V3": "V3 (PCA Component 3)",
        "V4": "V4 (PCA Component 4)",
        "V5": "V5 (PCA Component 5)",
        "V6": "V6 (PCA Component 6)",
        "V7": "V7 (PCA Component 7)",
        "V8": "V8 (PCA Component 8)",
        "V9": "V9 (PCA Component 9)",
        "V10": "V10 (PCA Component 10)",
        "V11": "V11 (PCA Component 11)",
        "V12": "V12 (PCA Component 12)",
        "V13": "V13 (PCA Component 13)",
        "V14": "V14 (PCA Component 14)",
        "V15": "V15 (PCA Component 15)",
        "V16": "V16 (PCA Component 16)",
        "V17": "V17 (PCA Component 17)",
        "V18": "V18 (PCA Component 18)",
        "V19": "V19 (PCA Component 19)",
        "V20": "V20 (PCA Component 20)",
        "V21": "V21 (PCA Component 21)",
        "V22": "V22 (PCA Component 22)",
        "V23": "V23 (PCA Component 23)",
        "V24": "V24 (PCA Component 24)",
        "V25": "V25 (PCA Component 25)",
        "V26": "V26 (PCA Component 26)",
        "V27": "V27 (PCA Component 27)",
        "V28": "V28 (PCA Component 28)",
        "Amount": "Transaction Amount ($)",
    }
