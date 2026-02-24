# XGBoost & LightGBM

> The undisputed champions of tabular data — XGBoost and LightGBM dominate Kaggle competitions and production ML systems alike.

## What Makes Them Special?

Both are **gradient boosting** frameworks that build trees sequentially, with each new tree correcting the errors of its predecessors. They improve on scikit-learn's `GradientBoostingClassifier` with:

- **Speed:** Histogram-based splitting and parallel processing make them orders of magnitude faster.
- **Regularisation:** Built-in L1/L2 penalties on leaf weights prevent overfitting.
- **Missing value handling:** Both handle `NaN` values natively without imputation.
- **Early stopping:** Training halts automatically when validation performance stops improving.

## XGBoost

```python
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
print(f"XGBoost Accuracy: {xgb.score(X_te, y_te):.3f}")
```

## LightGBM

LightGBM uses a **leaf-wise** growth strategy (rather than level-wise), which often converges faster and produces better accuracy on large datasets.

```python
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)
lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
print(f"LightGBM Accuracy: {lgb.score(X_te, y_te):.3f}")
```

## XGBoost vs LightGBM

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Tree growth | Level-wise | Leaf-wise (faster convergence) |
| Speed | Fast | Faster on large datasets |
| Categorical support | Requires encoding | Native categorical support |
| Community | Larger, more mature | Growing rapidly |

## Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | Number of boosting rounds (trees) |
| `learning_rate` | Step size for each tree's contribution — lower values need more trees |
| `max_depth` | Maximum tree depth — controls complexity |
| `subsample` | Fraction of rows used per tree — adds randomness, reduces overfitting |
| `colsample_bytree` | Fraction of features used per tree |
| `reg_alpha` / `reg_lambda` | L1 / L2 regularisation on leaf weights |

!!! tip "Workplace Tip"
    Start with `learning_rate=0.1`, `max_depth=5`, and `n_estimators=200` with early stopping. Then use `Optuna` or `GridSearchCV` to fine-tune. This combination wins more competitions than any other approach on tabular data.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Deploying state-of-the-art gradient boosting frameworks |
