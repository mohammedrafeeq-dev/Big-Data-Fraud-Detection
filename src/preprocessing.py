# ============================================================
# preprocessing.py — Data Cleaning & Preprocessing
# ============================================================
"""
Handles data cleaning, encoding, scaling, and PySpark ML Pipeline
construction for the fraud detection dataset.
"""

from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler

try:
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Paths, DataConfig, ModelConfig
    from src.utils import logger, timer


@timer
def drop_irrelevant_columns(df: DataFrame) -> DataFrame:
    """Drop columns not useful for modeling (e.g., id)."""
    cols_to_drop = [DataConfig.ID_COL]
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(*existing)
        logger.info(f"  Dropped columns: {existing}")
    return df


@timer
def handle_missing_values(df: DataFrame) -> DataFrame:
    """
    Handle missing values using median imputation for numeric columns.
    PySpark-optimized approach using approxQuantile for large datasets.
    """
    null_cols = []
    total = df.count()
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            null_cols.append((col_name, null_count))

    if not null_cols:
        logger.info("  No missing values found.")
        return df

    logger.info(f"  Imputing {len(null_cols)} columns with missing values...")
    for col_name, count in null_cols:
        median = df.approxQuantile(col_name, [0.5], 0.01)[0]
        df = df.fillna({col_name: median})
        logger.info(f"    {col_name}: {count} nulls → imputed with median={median:.4f}")

    return df


@timer
def remove_duplicates(df: DataFrame) -> DataFrame:
    """Remove duplicate rows from the DataFrame."""
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    removed = before - after
    logger.info(f"  Removed {removed:,} duplicate rows ({before:,} → {after:,})")
    return df


@timer
def handle_outliers(df: DataFrame, col: str = "Amount", method: str = "iqr") -> DataFrame:
    """
    Handle outliers using IQR method on the specified column.
    Caps values at Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    """
    quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before_outliers = df.filter(
        (F.col(col) < lower) | (F.col(col) > upper)
    ).count()

    df = df.withColumn(
        col,
        F.when(F.col(col) < lower, lower)
        .when(F.col(col) > upper, upper)
        .otherwise(F.col(col))
    )

    logger.info(f"  Outliers capped in '{col}': {before_outliers:,} values "
                f"(bounds: [{lower:.2f}, {upper:.2f}])")
    return df


@timer
def build_feature_vector(df: DataFrame, input_cols: list = None,
                         output_col: str = None) -> DataFrame:
    """
    Assemble feature columns into a single vector using VectorAssembler.
    """
    input_cols = input_cols or DataConfig.ALL_FEATURES
    output_col = output_col or DataConfig.FEATURES_COL

    # Only use columns that exist in the DataFrame
    available_cols = [c for c in input_cols if c in df.columns]
    logger.info(f"  Assembling {len(available_cols)} features into '{output_col}'")

    assembler = VectorAssembler(
        inputCols=available_cols,
        outputCol=output_col,
        handleInvalid="skip"
    )
    return assembler.transform(df)


@timer
def scale_features(df: DataFrame, input_col: str = None,
                   output_col: str = None) -> DataFrame:
    """Apply StandardScaler to the feature vector."""
    input_col = input_col or DataConfig.FEATURES_COL
    output_col = output_col or DataConfig.SCALED_FEATURES_COL

    scaler = StandardScaler(
        inputCol=input_col,
        outputCol=output_col,
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    logger.info(f"  Features scaled: {input_col} → {output_col}")
    return df


@timer
def prepare_label(df: DataFrame) -> DataFrame:
    """Ensure the target column is properly typed as a double label."""
    df = df.withColumn(DataConfig.LABEL_COL, F.col(DataConfig.TARGET_COL).cast("double"))
    logger.info(f"  Label column created: '{DataConfig.LABEL_COL}' from '{DataConfig.TARGET_COL}'")
    return df


@timer
def split_data(df: DataFrame, train_ratio: float = None, seed: int = None):
    """
    Split data into training and test sets.
    Returns (train_df, test_df).
    """
    train_ratio = train_ratio or ModelConfig.TRAIN_RATIO
    test_ratio = 1.0 - train_ratio
    seed = seed or ModelConfig.RANDOM_SEED

    train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)

    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"  Train: {train_count:,} | Test: {test_count:,} "
                f"(ratio: {train_ratio:.0%}/{test_ratio:.0%})")
    return train_df, test_df


@timer
def run_full_preprocessing(df: DataFrame) -> tuple:
    """
    Execute the complete preprocessing pipeline.
    Returns (train_df, test_df) ready for model training.
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Handle outliers on Amount
    df = handle_outliers(df, col="Amount")

    # Step 5: Prepare label
    df = prepare_label(df)

    # Step 6: Feature engineering is done in feature_engineering.py
    # (called before this in the pipeline)

    # Step 7: Build feature vector
    df = build_feature_vector(df)

    # Step 8: Scale features
    df = scale_features(df)

    # Step 9: Split data
    train_df, test_df = split_data(df)

    logger.info("Preprocessing pipeline complete!")
    return train_df, test_df
