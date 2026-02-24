# Scikit-Learn Regressors Reference

> Quick lookup for commonly utilized continuous machine learning algorithms within `sklearn`.

## Linear Models

### `LinearRegression`
**Use Case:** Baseline continuous relationships.
**Key Parameters:** None (Standard Ordinary Least Squares).

### `Ridge` and `Lasso`
**Use Case:** Regularised linear regression to drastically prevent overfitting.
**Key Parameters:**
* `alpha`: Regularisation strength (Larger means stronger penalty).

## Tree-Based Models

### `DecisionTreeRegressor`
**Use Case:** Non-linear rule-based regression.
**Key Parameters:**
* `max_depth`: Limits geometric tree depth.

### `RandomForestRegressor`
**Use Case:** Averaged, robust ensemble predictions.
**Key Parameters:**
* `n_estimators`: Count of trees constructed.
* `max_features`: Maximum variables considered at each dynamic split.

### `GradientBoostingRegressor`
**Use Case:** Corrective step-wise ensemble.
**Key Parameters:**
* `learning_rate`: Shrinkage parameter.
* `loss`: Loss function to minimize (e.g., `'squared_error'`).

## Advanced Models

### `SVR` (Support Vector Regressor)
**Use Case:** Geometric tube boundary analysis.
**Key Parameters:**
* `kernel`: Space transformation (`'linear'`, `'rbf'`).
* `C`: Margin strictness.
* `epsilon`: The mathematical width of the no-penalty tube.

## KSB Mapping

| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
