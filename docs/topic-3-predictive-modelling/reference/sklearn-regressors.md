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

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
