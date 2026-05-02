# ============================================================
# data_ingestion.py — Data Loading & Schema Enforcement
# ============================================================
"""
Handles loading raw data into PySpark DataFrames with explicit
schema enforcement, validation, and Parquet conversion.
Includes a synthetic data generator as fallback.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql import functions as F

try:
    from src.config import Paths, DataConfig
    from src.utils import logger, timer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Paths, DataConfig
    from src.utils import logger, timer


def get_schema() -> StructType:
    """Define explicit schema for the credit card fraud dataset."""
    fields = [StructField("id", IntegerType(), True)]
    for i in range(1, 29):
        fields.append(StructField(f"V{i}", DoubleType(), True))
    fields.append(StructField("Amount", DoubleType(), True))
    fields.append(StructField("Class", IntegerType(), True))
    return StructType(fields)


@timer
def generate_synthetic_data(num_rows=None, fraud_ratio=None, output_path=None):
    """
    Generate realistic synthetic credit card fraud dataset.
    Mimics statistical properties of the real Kaggle dataset.
    """
    num_rows = num_rows or DataConfig.SYNTHETIC_NUM_ROWS
    fraud_ratio = fraud_ratio or DataConfig.SYNTHETIC_FRAUD_RATIO
    output_path = Path(output_path or Paths.RAW_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic dataset: {num_rows:,} rows, {fraud_ratio:.1%} fraud")
    np.random.seed(42)

    num_fraud = int(num_rows * fraud_ratio)
    num_legit = num_rows - num_fraud

    # Legitimate transactions
    legit = {}
    for i in range(1, 29):
        legit[f"V{i}"] = np.random.normal(np.random.uniform(-0.5, 0.5),
                                           np.random.uniform(0.8, 2.5), num_legit)
    legit["Amount"] = np.clip(np.random.lognormal(3.5, 1.8, num_legit), 0, 25000)
    legit["Class"] = np.zeros(num_legit, dtype=int)

    # Fraudulent transactions — shifted distributions on key features
    fraud = {}
    for i in range(1, 29):
        if i in [1, 3, 4, 7, 10, 12, 14, 17]:
            mean, std = np.random.uniform(-3, -1), np.random.uniform(1.5, 4)
        elif i in [2, 5, 11, 16, 21]:
            mean, std = np.random.uniform(1, 3), np.random.uniform(1, 3)
        else:
            mean, std = np.random.uniform(-0.5, 0.5), np.random.uniform(1, 2.5)
        fraud[f"V{i}"] = np.random.normal(mean, std, num_fraud)
    fraud["Amount"] = np.clip(np.random.lognormal(4.5, 2.0, num_fraud), 0, 25000)
    fraud["Class"] = np.ones(num_fraud, dtype=int)

    # Combine, shuffle, save
    full_df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_df.insert(0, "id", range(len(full_df)))
    full_df["Amount"] = full_df["Amount"].round(2)
    full_df.to_csv(str(output_path), index=False)

    logger.info(f"Synthetic dataset saved: {output_path}")
    logger.info(f"  Total: {len(full_df):,} | Legit: {num_legit:,} | Fraud: {num_fraud:,}")
    return str(output_path)


@timer
def load_raw_data(spark: SparkSession, filepath=None) -> DataFrame:
    """Load raw CSV into Spark DataFrame. Auto-generates data if missing."""
    filepath = Path(filepath or Paths.RAW_CSV)
    if not filepath.exists():
        logger.warning(f"Raw data not found at {filepath}")
        generate_synthetic_data(output_path=filepath)

    logger.info(f"Loading data from: {filepath}")
    df = spark.read.option("header", "true").schema(get_schema()).csv(str(filepath))
    logger.info(f"Data loaded: {df.count():,} rows x {len(df.columns)} columns")
    return df


@timer
def validate_data(df: DataFrame) -> dict:
    """Perform comprehensive data quality checks."""
    logger.info("Running data validation checks...")
    total_rows = df.count()

    # Null analysis
    null_counts = {}
    for col_name in df.columns:
        nc = df.filter(F.col(col_name).isNull()).count()
        if nc > 0:
            null_counts[col_name] = nc

    # Duplicate analysis
    feature_cols = [c for c in df.columns if c != DataConfig.ID_COL]
    duplicates = total_rows - df.select(feature_cols).distinct().count()

    # Class distribution
    class_dist = df.groupBy(DataConfig.TARGET_COL).count().collect()
    class_distribution = {}
    for row in class_dist:
        label = "Legitimate" if row[DataConfig.TARGET_COL] == 0 else "Fraudulent"
        class_distribution[label] = {
            "count": row["count"],
            "percentage": round(row["count"] / total_rows * 100, 2),
        }

    results = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "null_columns": null_counts,
        "total_nulls": sum(null_counts.values()),
        "duplicates": duplicates,
        "class_distribution": class_distribution,
        "columns": df.columns,
        "dtypes": {f.name: str(f.dataType) for f in df.schema.fields},
    }

    logger.info(f"  Rows: {total_rows:,} | Cols: {len(df.columns)} | "
                f"Nulls: {results['total_nulls']} | Dupes: {duplicates}")
    for label, info in class_distribution.items():
        logger.info(f"  {label}: {info['count']:,} ({info['percentage']}%)")
    return results


@timer
def save_to_parquet(df: DataFrame, output_path=None):
    """Save DataFrame as Parquet for optimized downstream processing."""
    output_path = Path(output_path or Paths.PROCESSED_PARQUET)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output_path))
    logger.info(f"Saved Parquet: {output_path}")
