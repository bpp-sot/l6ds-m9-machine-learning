# Hyperparameter Cheatsheet

> Quick-reference table of the most important hyperparameters for common algorithms.

## Decision Trees

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_depth` | `None` (unbounded) | Maximum tree depth — lower values reduce overfitting |
| `min_samples_split` | `2` | Minimum samples to split a node — higher values reduce overfitting |
| `min_samples_leaf` | `1` | Minimum samples in a leaf — higher values smooth predictions |
| `ccp_alpha` | `0.0` | Cost-complexity pruning threshold — higher values prune more aggressively |

## Random Forest

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | `100` | Number of trees — more trees = more stable but slower |
| `max_features` | `'sqrt'` | Features per split — lower values add more randomness |
| `max_depth` | `None` | Same as Decision Tree |
| `min_samples_leaf` | `1` | Same as Decision Tree |

## Gradient Boosting / XGBoost / LightGBM

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | `100` | Number of boosting rounds |
| `learning_rate` | `0.1` | Step size — lower values need more estimators |
| `max_depth` | `3–6` | Tree depth per stage — lower values reduce overfitting |
| `subsample` | `1.0` | Row fraction per tree — values < 1 add stochasticity |
| `colsample_bytree` | `1.0` | Feature fraction per tree |
| `reg_alpha` / `reg_lambda` | `0` | L1 / L2 regularisation on leaf weights |

## SVM

| Parameter | Default | Effect |
|-----------|---------|--------|
| `C` | `1.0` | Regularisation — smaller C = wider margin, more misclassifications allowed |
| `gamma` | `'scale'` | RBF kernel reach — higher values = tighter boundaries |
| `kernel` | `'rbf'` | Kernel function: `linear`, `rbf`, `poly` |

## k-NN

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_neighbors` | `5` | Number of neighbours — lower k = more complex boundary |
| `weights` | `'uniform'` | `'uniform'` or `'distance'` — distance-weighted votes |
| `metric` | `'minkowski'` | Distance metric |

## Logistic Regression

| Parameter | Default | Effect |
|-----------|---------|--------|
| `C` | `1.0` | Inverse regularisation — smaller C = stronger penalty |
| `penalty` | `'l2'` | `'l1'`, `'l2'`, `'elasticnet'`, or `'none'` |
| `solver` | `'lbfgs'` | Optimiser — use `'saga'` for L1 or elasticnet |

!!! tip "Workplace Tip"
    Start with defaults. Only tune when you have a solid baseline and cross-validation pipeline in place.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
