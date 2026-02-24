# Optuna — Bayesian Hyperparameter Optimisation

> Optuna is a modern, Bayesian optimisation framework that intelligently searches the hyperparameter space — far more efficient than Grid or Random Search.

## Why Optuna?

- **Bayesian optimisation:** Learns from previous trials to focus on promising regions.
- **Pruning:** Automatically stops unpromising trials early.
- **Flexible:** Works with any ML framework (scikit-learn, XGBoost, LightGBM, PyTorch).

## Installation

```bash
pip install optuna
```

## Implementation

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)


def objective(trial):
    """Optuna objective function — returns the metric to optimise."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    return score


# Run optimisation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best Accuracy: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")
```

## With XGBoost

```python
from xgboost import XGBClassifier


def xgb_objective(trial):
    """Optimise XGBoost hyperparameters."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = XGBClassifier(**params, random_state=42, eval_metric="logloss")
    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    return score


study = optuna.create_study(direction="maximize")
study.optimize(xgb_objective, n_trials=50)
```

## Visualisation

```python
# Plot optimisation history
optuna.visualization.plot_optimization_history(study)

# Plot parameter importance
optuna.visualization.plot_param_importances(study)
```

!!! tip "Workplace Tip"
    Optuna with 50–100 trials typically outperforms `GridSearchCV` with thousands of combinations. Use it as your default tuning strategy for production models.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Advanced hyperparameter optimisation with Bayesian methods |
