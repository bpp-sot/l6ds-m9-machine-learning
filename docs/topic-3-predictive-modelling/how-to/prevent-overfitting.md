# How-to: Prevent Overfitting (Regularisation)

## The Problem
Your Random Forest achieves 99% accuracy on your Training Set, but drops to 65% when exposed to the validation holdout Set. You have memorized the training data's noise rather than the underlying pattern.

## The Solution

You must apply **Regularisation**—a deliberate, mathematical constraint that forces the model to learn a simpler, generalized pattern.

### 1. Regularisation in Linear Models (Ridge & Lasso)

By default, an OLS Linear Regression has no constraint. It will swing wildly to hit every point. `Ridge` (L2) and `Lasso` (L1) regressions apply a penalty factor, $\alpha$, which pushes the calculated coefficients towards zero.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=10, noise=20, random_state=42)

# Unregulated
ols = LinearRegression().fit(X, y)
print(f"OLS Training Variance: {np.var(ols.coef_):.2f}")

# Ridge (L2 Penalty) pushes all coefficients towards zero, shrinking them
ridge = Ridge(alpha=10.0).fit(X, y)
print(f"Ridge Training Variance: {np.var(ridge.coef_):.2f}")

# Lasso (L1 Penalty) pushes weak coefficients EXACTLY to zero
lasso = Lasso(alpha=5.0).fit(X, y)
print(f"Lasso Coefficients: {lasso.coef_}")
```

### 2. Regularisation in Tree Models

Decision Trees naturally overfit because they split infinitely until they reach pure leaves. We constrain them using structural pruning parameters.

```python
from sklearn.ensemble import RandomForestClassifier

# OVERFIT TREE: Will build until every leaf has 1 sample
bad_rf = RandomForestClassifier(max_depth=None, min_samples_split=2)

# REGULARIZED TREE:
good_rf = RandomForestClassifier(
    max_depth=5,            # Limit vertical depth
    min_samples_split=20,   # A node MUST have 20 samples to split again
    min_samples_leaf=10,    # A leaf MUST contain at least 10 samples
    max_features='sqrt'     # Only look at a random fraction of columns per split
)
```

### 3. Early Stopping in Gradient Boosting

Gradient Boosters iteratively chase errors. If you let an XGBoost model build 10,000 trees, it will perfectly memorize the dataset. We use **Early Stopping** to halt execution the exact moment the validation score starts getting worse.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Initialize DMatrix (XGBoost specific data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1}

# The model will train for up to 1000 rounds...
# BUT, if the validation metric (rmse) doesn't improve for 10 consecutive rounds, it aborts.
model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=1000, 
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=False
)

print(f"Best Iteration: {model.best_iteration}")
```

## Discussion

Applying regularization usually *decreases* your Training Score. This is a good thing. You are trading a perfect training score for a structurally resilient Testing Score.
