# Reference: Scikit-Learn Feature Selection API

This page organizes the primary `sklearn.feature_selection` classes and functions.

## 1. Filter Methods (Univariate)

Filter methods execute statistically independent of any machine learning model.

| Class | Purpose | Key Parameters | Note |
|-------|---------|----------------|------|
| `VarianceThreshold(threshold)` | Drops features where empirical variance falls below threshold | `threshold=0` drops purely static columns | Excellent first-pass sanity check. |
| `SelectKBest(score_func, k)` | Selects the `k` features with highest statistical scores | `k=10` keeps top 10 features | Must pair with appropriate `score_func`. |
| `SelectPercentile(score_func, percentile)`| Selects the top `percentile` percentage of features | `percentile=10` keeps top 10% | Dynamic `k` based on total dimension count. |

### Statistical Testing Functions (Used inside `score_func`)

| Function | Task Type | Feature Type | Logic |
|----------|-----------|--------------|-------|
| `f_regression` | Regression | Continuous | ANOVA F-Value (Linear Correlation) |
| `mutual_info_regression` | Regression | Any | Captures non-linear relationships |
| `f_classif` | Classification | Continuous | ANOVA F-Value (Linear Correlation) |
| `chi2` | Classification | Categorical | Chi-Squared test (strictly Non-negative data) |
| `mutual_info_classif` | Classification | Any | Captures non-linear relationships |

---

## 2. Wrapper Methods 

Wrapper methods iteratively train a user-provided estimator model.

| Class | Purpose | Requirements |
|-------|---------|--------------|
| `RFE(estimator, n_features_to_select)` | Recursive Feature Elimination. Iteratively drops the weakest feature. | `estimator` must expose `coef_` or `feature_importances_`. |
| `RFECV(estimator, cv, scoring)` | Automated RFE using Cross-Validation to find the mathematical optimal feature count. | Visually plotted via `.cv_results_`. |
| `SequentialFeatureSelector(estimator, direction)`| Adds (Forward) or removes (Backward) features one by one evaluating CV score shifts. | Compatible with models lacking importances natively (e.g. KNN). |

---

## 3. Embedded Methods

Embedded methods select features during the natural model training process.

| Algorithm Category | Built-in Feature Selection Logic | Extracted Attribute |
|--------------------|----------------------------------|---------------------|
| `Lasso` (L1 Regression) | Punishes magnitude of weights; forces weak features to exactly `0.0`. | `lasso.coef_` |
| Tree / Ensembles | Chooses features that produce highest Gini Impurity reduction / Information Gain at split nodes. | `rf.feature_importances_` |

To functionally drop columns using embedded logic inside an automated Pipeline, wrap the estimator:
- `SelectFromModel(estimator, threshold)`: Retains columns whose `coef_` or `importances_` exceed the defined threshold (e.g., `'median'`, `'1.5*mean'`).
