# Reference: Scikit-Learn Model Cheatsheet

This reference provides syntax lookups for initializing the foundational Machine Learning algorithms within Scikit-Learn.

## Generalized Linear Models (Parametric)

> **Core Concept:** Compute weights ($\beta$) via gradient tracking. Mathematically rigid. Fast computation but High Bias constraints. **Requires scaling.**

| Class | Type | Syntax | Core Parameters |
|-------|------|--------|-----------------|
| `LinearRegression` | Reg. | `from sklearn.linear_model import LinearRegression` | N/A |
| `Ridge` | Reg. | `from sklearn.linear_model import Ridge` | `alpha=1.0` (L2 Penalty) |
| `Lasso` | Reg. | `from sklearn.linear_model import Lasso` | `alpha=1.0` (L1 Penalty) |
| `LogisticRegression` | Class. | `from sklearn.linear_model import LogisticRegression`|  `C=1.0` (Inverse regularization penalty) |

## Distance-Based Models

> **Core Concept:** Measure spatial proximity of vector dimensions to establish relationships. Highly volatile to outliers. **Requires scaling.**

| Class | Type | Syntax | Core Parameters |
|-------|------|--------|-----------------|
| `KNeighborsClassifier` | Class. | `from sklearn.neighbors import KNeighborsClassifier` | `n_neighbors=5`, `weights='uniform'` |
| `SVC` | Class. | `from sklearn.svm import SVC` | `kernel='rbf'`, `C=1.0`, `gamma='scale'` |
| `SVR` | Reg. | `from sklearn.svm import SVR` | `epsilon=0.1` (Margin of tolerance) |

## Tree-Based Ensembles (Non-Parametric)

> **Core Concept:** Split multi-dimensional space via threshold conditions dynamically mapped to Information Gain or Gini index changes. Highly robust, structurally chaotic. **Scaling optional.**

| Class | Type | Syntax | Core Parameters |
|-------|------|--------|-----------------|
| `DecisionTreeClassifier`| Class. | `from sklearn.tree import DecisionTreeClassifier`| `max_depth`, `min_samples_split` |
| `RandomForestRegressor` | Reg. | `from sklearn.ensemble import RandomForestRegressor`| `n_estimators`, `max_features` |
| `GradientBoostingClassifier`| Class. | `from sklearn.ensemble import GradientBoostingClassifier`| `learning_rate` (gamma), `n_estimators` |

*(Note: While `GradientBoostingClassifier` exists natively in Sklearn, industry standard usually replaces this explicitly with XGBoost, LightGBM, or CatBoost).*

## Neural Architecture

> **Core Concept:** Mimics biological cognitive layers via activation functions multiplying iterative node signals. Requires immense hardware optimization. **Requires scaling.**

| Class | Type | Syntax | Core Parameters |
|-------|------|--------|-----------------|
| `MLPClassifier` | Class. | `from sklearn.neural_network import MLPClassifier`| `hidden_layer_sizes`, `activation`, `solver`, `learning_rate_init` |
