# ============================================================
# utils.py - Shared Utilities
# ============================================================
"""
Utility functions used across the fraud detection pipeline.
Includes logging setup, timing decorators, display helpers,
and file I/O utilities.
"""

import os
import sys
import time
import json
import logging
import functools
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# Logging Configuration
# ============================================================
def setup_logger(name: str = "BigDataProject", level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured logger with console output.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with formatted output
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger()


# ============================================================
# Timer Decorator
# ============================================================
def timer(func):
    """
    Decorator that logs the execution time of a function.

    Usage
    -----
        @timer
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[START] {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        # Format elapsed time nicely
        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"

        logger.info(f"[DONE]  {func.__name__} ({time_str})")
        return result
    return wrapper


# ============================================================
# Display Helpers
# ============================================================
def print_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    border = "=" * width
    padding = " " * ((width - len(title)) // 2)
    print(f"\n{border}")
    print(f"{padding}{title}")
    print(f"{border}\n")


def print_separator(char: str = "-", width: int = 60) -> None:
    """Print a simple line separator."""
    print(char * width)


def print_metric(name: str, value, fmt: str = ".4f") -> None:
    """Print a single metric with formatting."""
    if isinstance(value, float):
        print(f"  [*] {name:<30} : {value:{fmt}}")
    else:
        print(f"  [*] {name:<30} : {value}")


# ============================================================
# File I/O Helpers
# ============================================================
def save_json(data: dict, filepath: Path) -> None:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialize.
    filepath : Path
        Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved: {filepath}")


def load_json(filepath: Path) -> dict:
    """
    Load a JSON file as a dictionary.

    Parameters
    ----------
    filepath : Path
        Input file path.

    Returns
    -------
    dict
        Deserialized data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# DataFrame Display Utility
# ============================================================
def show_spark_df_info(df, name: str = "DataFrame") -> None:
    """
    Print summary information about a Spark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame to summarize.
    name : str
        Display name for the DataFrame.
    """
    print_header(f"{name} Summary")
    print(f"  Rows    : {df.count():,}")
    print(f"  Columns : {len(df.columns)}")
    print(f"  Schema  :")
    for field in df.schema.fields:
        print(f"    - {field.name:<20} {str(field.dataType):<20} nullable={field.nullable}")
    print()
