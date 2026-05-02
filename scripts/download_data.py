# ============================================================
# download_data.py — Data Acquisition Script
# ============================================================
"""
Downloads the credit card fraud detection dataset from Kaggle.
Falls back to synthetic data generation if Kaggle API is unavailable.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DataConfig
from src.utils import logger
from src.data_ingestion import generate_synthetic_data


def download_from_kaggle():
    """Attempt to download dataset via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        Paths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading: {DataConfig.KAGGLE_DATASET}")
        api.dataset_download_files(
            DataConfig.KAGGLE_DATASET,
            path=str(Paths.RAW_DATA_DIR),
            unzip=True,
        )

        # Find the downloaded CSV
        csv_files = list(Paths.RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            # Rename to our standard name if needed
            downloaded = csv_files[0]
            if downloaded.name != Paths.RAW_CSV.name:
                downloaded.rename(Paths.RAW_CSV)
            logger.info(f"Dataset downloaded: {Paths.RAW_CSV}")
            return True

    except ImportError:
        logger.warning("Kaggle library not installed. pip install kaggle")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")

    return False


def main():
    """Main download logic with fallback."""
    logger.info("=" * 60)
    logger.info("DATA ACQUISITION")
    logger.info("=" * 60)

    if Paths.RAW_CSV.exists():
        logger.info(f"Dataset already exists: {Paths.RAW_CSV}")
        return

    # Try Kaggle first
    success = download_from_kaggle()

    if not success:
        logger.info("Falling back to synthetic data generation...")
        generate_synthetic_data()

    logger.info("Data acquisition complete!")


if __name__ == "__main__":
    main()
