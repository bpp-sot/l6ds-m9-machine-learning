# Environment Setup

## Quick Start with Google Colab

The fastest way to start is [Google Colab](https://colab.research.google.com/) — no installation required. All libraries are pre-installed.

## Local Installation

### Step 1: Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/).

### Step 2: Create a Virtual Environment

```bash
python -m venv ml-env
# Windows
ml-env\\Scripts\\activate
# Mac/Linux
source ml-env/bin/activate
```

### Step 3: Install Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm
pip install shap lime
pip install optuna
pip install statsmodels prophet
pip install jupyter
```

### Step 4: Launch Jupyter

```bash
jupyter notebook
```

## Library Usage Map

| Library | Used For | Topics |
|---------|----------|--------|
| `pandas` | Data loading, manipulation, cleaning | All |
| `numpy` | Numerical operations | All |
| `matplotlib` / `seaborn` | Visualisation | All |
| `scikit-learn` | ML algorithms, preprocessing, evaluation | 1–5, 7 |
| `xgboost` / `lightgbm` | Gradient boosting | 3, 4 |
| `statsmodels` | Time series analysis, statistical tests | 6 |
| `prophet` | Automated time series forecasting | 6 |
| `shap` / `lime` | Model interpretability | 8 |
| `optuna` | Bayesian hyperparameter optimisation | 7 |

!!! tip "Workplace Tip"
    Keep a `requirements.txt` in your project so colleagues can reproduce your environment: `pip freeze > requirements.txt`
