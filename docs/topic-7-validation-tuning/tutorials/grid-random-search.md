# Grid Search & Random Search

> Hyperparameter tuning is the process of finding the best configuration for your model. Grid Search and Random Search are the two foundational approaches.

## Grid Search (Exhaustive)

`GridSearchCV` tries **every combination** in the parameter grid. It is thorough but expensive — a grid with 4 parameters × 5 values each tests 625 combinations.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")
```

## Random Search (Efficient)

`RandomizedSearchCV` samples a fixed number of random combinations from parameter distributions. It is faster and often finds equally good results.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,          # Try 50 random combinations
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
random_search.fit(X, y)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV Accuracy: {random_search.best_score_:.4f}")
```

## Grid vs Random

| Aspect | Grid Search | Random Search |
|--------|-------------|---------------|
| Coverage | Exhaustive | Sampled |
| Speed | Slow for large grids | Tuneable via `n_iter` |
| Best for | Small, focused grids | Large parameter spaces |
| Discovery | Only tests specified values | Can discover unexpected optima |

!!! tip "Workplace Tip"
    Start with `RandomizedSearchCV(n_iter=50)` to identify promising regions, then refine with a focused `GridSearchCV` around the best values.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Systematic hyperparameter optimisation |
