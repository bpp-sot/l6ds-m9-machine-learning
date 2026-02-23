# Scikit-Learn Feature Selection API

> The most frequently natively utilized dimensionality optimization modules computationally located within sklearn.

## Statistical Filters 

Evaluating vectors fundamentally strictly via mathematical boundaries computationally independently.

| Class | Methodology | Scoring Function Support (Target) |
|---|---|---|
| `VarianceThreshold(threshold=0)` | Deletes identical fixed columns structurally | None required (Unsupervised) |
| `SelectKBest(score_func=...)` | Keeps strictly top `K` numeric arrays independently | Requires Regression/Classification scoring explicitly |
| `SelectPercentile(percentile=10)` | Retains exactly the top `X%` distributions geometrically | Requires Regression/Classification scoring explicitly |

### Scoring Functions for `SelectKBest`

| Score Func | Predictor Vector (`X`) | Target Vector (`Y`) | Use Case |
|---|---|---|---|
| `f_classif` | Continuous Float | Categorical Binary | **Classification** |
| `chi2` | Categorical Integer | Categorical Binary | **Classification** |
| `f_regression` | Continuous Float | Continuous Float | **Regression** |
| `mutual_info_classif` | Non-Linear Float | Categorical Binary | **Classification** (Heavy compute) |

## Wrapper Constructors

Searching algorithmic arrays structurally combinatorially dynamically.

| Class | Description | Scaling Efficiency |
|---|---|---|
| `RFE(estimator=rf, step=1)` | Recursive Feature Elimination implicitly strips the singularly lowest performing matrix explicitly mechanically natively backwards | Medium |
| `RFECV(cv=5)` | RFE mechanically executed sequentially across 5 completely separated Validation distributions natively preventing extreme Data Leakage mathematically | Slow |
| `SequentialFeatureSelector()` | Forwards or backwards combinatorial explicit isolation matrix algebraically geometrically | Extremely Slow |

## Feature Importances (Embedded)

Extracting explicitly algorithmic constraints organically natively explicitly computational generated weights algebraically.

| Technique | Metric | How to Call |
|---|---|---|
| **Tree Ensembles** | Gini Impurity or Entropy Decreases strictly | `model.feature_importances_` |
| **Linear Models** | Absolute weight parameter size functionally globally | `model.coef_` |
| **Lasso Regression (L1)** | Feature Coefficient forced mechanically to exactly `0` | `model.coef_ == 0` |

!!! tip "Workplace Tip"
    To extract the physical column string matrix natively out from `SelectKBest`, execute `selector.get_support()`. This yields explicitly a boolean array internally `[True, False, True]`, allowing you to slice dynamically `df.columns[selector.get_support()]` elegantly algebraically.
