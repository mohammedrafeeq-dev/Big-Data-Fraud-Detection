# 🛡️ Credit Card Fraud Detection — Big Data Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/Apache_Spark-3.5+-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLOps](https://github.com/mohammedrafeeq-dev/Big-Data-Fraud-Detection/actions/workflows/train_models.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade, end-to-end Big Data pipeline for detecting fraudulent credit card transactions using Apache Spark MLlib.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Project Structure](#-project-structure) · [Results](#-results)

</div>

---

## 📋 Overview

This project implements a complete Big Data analytics pipeline that processes **500,000+ credit card transactions** to identify fraudulent activity. Built with **Apache Spark** for distributed processing and **PySpark MLlib** for scalable machine learning, the system trains and compares four classification models before deploying the best performer through an interactive **Streamlit** web application.

### Business Problem

Credit card fraud costs financial institutions billions of dollars annually. This system addresses the challenge of detecting fraudulent transactions in real-time from highly imbalanced datasets where fraud represents only ~1.7% of all transactions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **End-to-End Pipeline** | From data ingestion to deployed predictions |
| 📊 **Comprehensive EDA** | 7+ visualization types with business insights |
| 🧠 **4 ML Models** | Logistic Regression, Random Forest, Decision Tree, GBT |
| ⚡ **Scalable Processing** | PySpark DataFrames and MLlib pipelines |
| 🎨 **Modern UI** | Dark-themed Streamlit app with glassmorphism design |
| 📁 **Batch Prediction** | Upload CSV for multi-model bulk predictions |
| 📈 **Model Comparison** | Interactive charts, radar plots, and confusion matrices |
| 🔮 **Real-Time Prediction** | Instant fraud scoring with confidence levels |
| 🤖 **MLOps Automation** | GitHub Actions pipeline for automated model training |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│              Kaggle API / Synthetic Generator                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  DATA VALIDATION                            │
│         Schema Enforcement · Quality Checks                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              EXPLORATORY DATA ANALYSIS                      │
│      Statistics · Visualizations · Business Insights        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             FEATURE ENGINEERING                             │
│     Log Transform · Z-Score · Interactions · Binning        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                PREPROCESSING                                │
│   Cleaning · Scaling · VectorAssembly · Train/Test Split    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               MODEL TRAINING                                │
│    Logistic Reg · Random Forest · Decision Tree · GBT       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              MODEL EVALUATION                               │
│   Accuracy · Precision · Recall · F1 · AUC · Confusion     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 DEPLOYMENT                                  │
│          Streamlit Web Application (5 Pages)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Java 8, 11, or 17 (required for PySpark)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/big-data-fraud-detection.git
cd big-data-fraud-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### 🪟 Windows-Specific Setup (Important)

PySpark requires Hadoop binaries (`winutils.exe`) to perform file operations on Windows. If you encounter errors during model saving:

1. Create a folder `C:\hadoop\bin`.
2. Download `winutils.exe` (e.g., from [cdarlint/winutils](https://github.com/cdarlint/winutils)) and place it in the `bin` folder.
3. Set the environment variable `HADOOP_HOME` to `C:\hadoop`.
4. Add `%HADOOP_HOME%\bin` to your system `PATH`.

### Run the Pipeline

```bash
# Execute the complete end-to-end pipeline
python scripts/run_pipeline.py
```

This will:
1. Generate/load the dataset (500K+ transactions)
2. Run EDA and save visualizations
3. Engineer features and preprocess data
4. Train 4 classification models
5. Evaluate and compare all models
6. Save artifacts to `models/` and `outputs/`

### Launch the Web App

```bash
cd app
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Big Data Project/
├── data/
│   ├── raw/                        # Original dataset (CSV)
│   └── processed/                  # Cleaned data (Parquet)
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Central configuration
│   ├── spark_session.py            # Spark session factory
│   ├── data_ingestion.py           # Data loading & validation
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── preprocessing.py            # Cleaning & scaling pipeline
│   ├── feature_engineering.py      # Feature creation
│   ├── model_training.py           # MLlib model training
│   ├── model_evaluation.py         # Metrics & comparison
│   └── utils.py                    # Logging & helpers
├── models/
│   ├── logistic_regression/        # Saved LR model
│   ├── random_forest/              # Saved RF model
│   ├── decision_tree/              # Saved DT model
│   ├── gbt/                        # Saved GBT model
│   └── model_metrics.json          # All model metrics
├── app/
│   ├── streamlit_app.py            # Main Streamlit entry point
│   ├── pages/
│   │   ├── home.py                 # Project overview
│   │   ├── eda_dashboard.py        # EDA visualizations
│   │   ├── predict.py              # Real-time prediction
│   │   ├── model_comparison.py     # Model comparison
│   │   └── upload_dataset.py       # CSV upload & batch predict
│   └── .streamlit/config.toml      # Theme configuration
├── scripts/
│   ├── run_pipeline.py             # Master orchestration
│   └── download_data.py            # Data acquisition
├── outputs/
│   └── visualizations/             # Saved EDA charts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier with elastic net regularization |
| **Random Forest** | Ensemble of 100 decision trees with bagging |
| **Decision Tree** | Single interpretable tree classifier |
| **Gradient Boosted Trees** | Sequential ensemble with boosting |

All models are trained using **PySpark MLlib** with:
- StandardScaler for feature normalization
- VectorAssembler for feature vectorization
- Configurable hyperparameters via `src/config.py`

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Fraction of predicted frauds that are actual frauds |
| **Recall** | Fraction of actual frauds correctly detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |
| **Confusion Matrix** | TP, TN, FP, FN breakdown |

---

## 🛠️ Technology Stack

- **Apache Spark 3.5+** — Distributed data processing
- **PySpark MLlib** — Scalable machine learning
- **Streamlit** — Web application framework
- **Plotly** — Interactive visualizations
- **Seaborn / Matplotlib** — Statistical charts
- **Pandas / NumPy** — Data manipulation
- **scikit-learn** — Evaluation metrics

---

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">
    <p style="color: #888;">
        Built with ❤️ using Apache Spark & Streamlit
    </p>
</div>
