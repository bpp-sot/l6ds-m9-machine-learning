# Scikit-Learn Model Selection Reference

> Quick-reference for the `sklearn.model_selection` module — the toolkit for splitting, validating, and tuning models.

## Splitting

| Class / Function | Purpose |
|-----------------|---------|
| `train_test_split(X, y)` | Single random split into train and test sets |
| `StratifiedShuffleSplit` | Repeated random splits preserving class proportions |
| `TimeSeriesSplit` | Expanding-window splits for time-ordered data |

## Cross-Validation

| Function | Purpose |
|----------|---------|
| `cross_val_score(model, X, y, cv=5)` | Returns array of scores for each fold |
| `cross_validate(model, X, y, cv=5)` | Returns dict with fit time, score time, and test scores |
| `cross_val_predict(model, X, y, cv=5)` | Returns out-of-fold predictions for every sample |

### CV Splitters

| Splitter | Use Case |
|----------|----------|
| `KFold(n_splits=5)` | Standard k-Fold (regression) |
| `StratifiedKFold(n_splits=5)` | Preserves class ratios per fold (classification) |
| `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` | Repeated stratified for more stable estimates |
| `LeaveOneOut()` | Every sample used as a test set once — expensive |

## Hyperparameter Search

| Class | Strategy |
|-------|----------|
| `GridSearchCV` | Exhaustive search over a parameter grid |
| `RandomizedSearchCV` | Random sampling from parameter distributions |
| `HalvingGridSearchCV` | Successive halving — efficient for large grids |

## Quick Example

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")
```

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
