# ============================================================
# feature_engineering.py — Feature Creation & Transformations
# ============================================================
"""
Creates derived features from the raw dataset to improve
model performance. Includes log transformations, statistical
features, and interaction terms.
"""

from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

try:
    from src.config import DataConfig
    from src.utils import logger, timer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import DataConfig
    from src.utils import logger, timer


@timer
def add_log_amount(df: DataFrame) -> DataFrame:
    """
    Add log-transformed amount feature.
    log(Amount + 1) to handle zero amounts gracefully.
    """
    df = df.withColumn("log_amount", F.log1p(F.col("Amount")))
    logger.info("  Added feature: log_amount = log(Amount + 1)")
    return df


@timer
def add_amount_zscore(df: DataFrame) -> DataFrame:
    """
    Add z-score of transaction amount.
    Measures how many standard deviations each amount is from the mean.
    """
    stats = df.select(
        F.mean("Amount").alias("mean"),
        F.stddev("Amount").alias("std")
    ).collect()[0]

    mean_val = stats["mean"]
    std_val = stats["std"] if stats["std"] and stats["std"] > 0 else 1.0

    df = df.withColumn(
        "amount_zscore",
        (F.col("Amount") - F.lit(mean_val)) / F.lit(std_val)
    )
    logger.info(f"  Added feature: amount_zscore (mean={mean_val:.2f}, std={std_val:.2f})")
    return df


@timer
def add_feature_interactions(df: DataFrame) -> DataFrame:
    """
    Add interaction features between highly correlated PCA components.
    These capture non-linear relationships the models might miss.
    """
    # V1 * V3 — both are strong fraud indicators
    if "V1" in df.columns and "V3" in df.columns:
        df = df.withColumn("V1_V3_interaction", F.col("V1") * F.col("V3"))

    # V14 * V17 — highly discriminative features
    if "V14" in df.columns and "V17" in df.columns:
        df = df.withColumn("V14_V17_interaction", F.col("V14") * F.col("V17"))

    logger.info("  Added interaction features: V1_V3, V14_V17")
    return df


@timer
def add_amount_bin(df: DataFrame) -> DataFrame:
    """
    Categorize transaction amounts into bins.
    Useful for capturing non-linear amount patterns.
    """
    df = df.withColumn(
        "amount_bin",
        F.when(F.col("Amount") <= 10, 0.0)
        .when(F.col("Amount") <= 50, 1.0)
        .when(F.col("Amount") <= 200, 2.0)
        .when(F.col("Amount") <= 1000, 3.0)
        .otherwise(4.0)
    )
    logger.info("  Added feature: amount_bin (5 bins)")
    return df


@timer
def run_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Execute the complete feature engineering pipeline.

    New features added:
    - log_amount: Log-transformed transaction amount
    - amount_zscore: Z-score normalized amount
    - V1_V3_interaction: Interaction between V1 and V3
    - V14_V17_interaction: Interaction between V14 and V17
    - amount_bin: Binned transaction amount

    Parameters
    ----------
    df : DataFrame
        Input DataFrame after initial cleaning.

    Returns
    -------
    DataFrame
        DataFrame with engineered features appended.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    df = add_log_amount(df)
    df = add_amount_zscore(df)
    df = add_feature_interactions(df)
    df = add_amount_bin(df)

    logger.info(f"  Total columns after engineering: {len(df.columns)}")
    return df
