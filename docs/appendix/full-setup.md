# Full Setup Guide

> Everything you need installed to run the code in this module.

## Python Version

We recommend **Python 3.10+**. Check your version:

```bash
python --version
```

## Virtual Environment

Always use a virtual environment to isolate your project dependencies:

```bash
# Create the environment
python -m venv venv

# Activate it
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels xgboost lightgbm shap lime optuna prophet joblib imbalanced-learn gower kmodes pmdarima
```

## Core Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | ML algorithms, preprocessing, evaluation, model selection |
| `seaborn` / `matplotlib` | Data visualisation |
| `statsmodels` | Statistical models and time series analysis |
| `xgboost` / `lightgbm` | Gradient boosting frameworks |
| `shap` / `lime` | Model explainability |
| `optuna` | Bayesian hyperparameter optimisation |
| `prophet` | Automated time series forecasting |
| `joblib` | Saving and loading trained models |
| `imbalanced-learn` | SMOTE and other resampling techniques |

## Verify Installation

```python
import sklearn
import pandas as pd
import seaborn as sns

print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
print(f"seaborn: {sns.__version__}")
print("All core libraries installed successfully.")
```

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K1 | Context of Data Science | Understanding where ML sits within the broader discipline |
| S3 | Programming languages and tools | Setting up the development environment and dependencies |
| B6 | Commitment to keeping up to date | Engaging with current ML resources and research |
