# ============================================================
# model_training.py — MLlib Model Training Pipelines
# ============================================================
"""
Trains multiple PySpark MLlib classification models for fraud detection.
Supports Logistic Regression, Random Forest, Decision Tree, and GBT.
Each model is saved to disk for later inference.
"""

from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    DecisionTreeClassifier,
    GBTClassifier,
)

try:
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer


FEATURES_COL = DataConfig.SCALED_FEATURES_COL
LABEL_COL = DataConfig.LABEL_COL


@timer
def train_logistic_regression(train_df: DataFrame):
    """Train a Logistic Regression model with elastic net regularization."""
    logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=ModelConfig.LR_MAX_ITER,
        regParam=ModelConfig.LR_REG_PARAM,
        elasticNetParam=ModelConfig.LR_ELASTIC_NET,
        family="binomial",
    )
    model = lr.fit(train_df)
    logger.info(f"    Coefficients count: {len(model.coefficients)}")
    logger.info(f"    Intercept: {model.intercept:.4f}")
    return model


@timer
def train_random_forest(train_df: DataFrame):
    """Train a Random Forest Classifier."""
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        numTrees=ModelConfig.RF_NUM_TREES,
        maxDepth=ModelConfig.RF_MAX_DEPTH,
        maxBins=ModelConfig.RF_MAX_BINS,
        subsamplingRate=ModelConfig.RF_SUBSAMPLING_RATE,
        seed=ModelConfig.RANDOM_SEED,
    )
    model = rf.fit(train_df)
    logger.info(f"    Number of trees: {model.getNumTrees}")
    return model


@timer
def train_decision_tree(train_df: DataFrame):
    """Train a Decision Tree Classifier."""
    logger.info("  Training Decision Tree...")
    dt = DecisionTreeClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxDepth=ModelConfig.DT_MAX_DEPTH,
        maxBins=ModelConfig.DT_MAX_BINS,
        seed=ModelConfig.RANDOM_SEED,
    )
    model = dt.fit(train_df)
    logger.info(f"    Tree depth: {model.depth}")
    return model


@timer
def train_gbt(train_df: DataFrame):
    """Train a Gradient Boosted Trees Classifier."""
    logger.info("  Training Gradient Boosted Trees...")
    gbt = GBTClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=ModelConfig.GBT_MAX_ITER,
        maxDepth=ModelConfig.GBT_MAX_DEPTH,
        stepSize=ModelConfig.GBT_STEP_SIZE,
        subsamplingRate=ModelConfig.GBT_SUBSAMPLING_RATE,
        seed=ModelConfig.RANDOM_SEED,
    )
    model = gbt.fit(train_df)
    logger.info(f"    Number of trees: {model.getNumTrees}")
    return model


def save_model(model, path: Path):
    """Save a trained model to disk."""
    path = Path(path)
    if path.exists():
        import shutil
        shutil.rmtree(str(path))
    model.save(str(path))
    logger.info(f"  Model saved: {path}")


@timer
def train_all_models(train_df: DataFrame) -> dict:
    """
    Train all classification models and save them.

    Returns
    -------
    dict
        Dictionary mapping model keys to trained model objects.
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)

    # Cache training data for performance
    train_df.cache()
    train_count = train_df.count()
    logger.info(f"  Training set size: {train_count:,}")

    models = {}

    # 1. Logistic Regression
    try:
        models["lr"] = train_logistic_regression(train_df)
        save_model(models["lr"], Paths.LR_MODEL_DIR)
    except Exception as e:
        logger.error(f"  LR training failed: {e}")

    # 2. Random Forest
    try:
        models["rf"] = train_random_forest(train_df)
        save_model(models["rf"], Paths.RF_MODEL_DIR)
    except Exception as e:
        logger.error(f"  RF training failed: {e}")

    # 3. Decision Tree
    try:
        models["dt"] = train_decision_tree(train_df)
        save_model(models["dt"], Paths.DT_MODEL_DIR)
    except Exception as e:
        logger.error(f"  DT training failed: {e}")

    # 4. Gradient Boosted Trees
    try:
        models["gbt"] = train_gbt(train_df)
        save_model(models["gbt"], Paths.GBT_MODEL_DIR)
    except Exception as e:
        logger.error(f"  GBT training failed: {e}")

    train_df.unpersist()

    logger.info(f"  Successfully trained {len(models)}/{4} models")
    return models
