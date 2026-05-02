# ============================================================
# spark_session.py — Spark Session Factory
# ============================================================
"""
Creates and configures an optimized Apache Spark session for the
fraud detection pipeline. Includes performance tuning for local
execution with adaptive query execution and memory management.
"""

from pyspark.sql import SparkSession

# Import config — use try/except for flexibility when run from different dirs
try:
    from src.config import SparkConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import SparkConfig


def create_spark_session(app_name: str = None) -> SparkSession:
    """
    Create and return an optimized SparkSession.

    Parameters
    ----------
    app_name : str, optional
        Custom application name. Defaults to SparkConfig.APP_NAME.

    Returns
    -------
    SparkSession
        Configured Spark session ready for use.
    """
    name = app_name or SparkConfig.APP_NAME

    spark = (
        SparkSession.builder
        .appName(name)
        .master(SparkConfig.MASTER)
        # ── Memory Settings ──────────────────────────────────
        .config("spark.driver.memory", SparkConfig.DRIVER_MEMORY)
        .config("spark.executor.memory", SparkConfig.EXECUTOR_MEMORY)
        # ── Performance Tuning ───────────────────────────────
        .config("spark.sql.shuffle.partitions", SparkConfig.SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", SparkConfig.ADAPTIVE_ENABLED)
        .config("spark.serializer", SparkConfig.SERIALIZER)
        # ── SQL Settings ─────────────────────────────────────
        .config("spark.sql.legacy.timeParserPolicy", SparkConfig.LEGACY_TIME_PARSER)
        # ── UI / Logging ─────────────────────────────────────
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=WARN")
        .getOrCreate()
    )

    # Reduce log verbosity
    spark.sparkContext.setLogLevel("WARN")

    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """
    Gracefully stop a SparkSession.

    Parameters
    ----------
    spark : SparkSession
        The session to stop.
    """
    if spark is not None:
        spark.stop()
