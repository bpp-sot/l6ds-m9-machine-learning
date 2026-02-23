# Embedded Selection Methods

> "Why build a separate feature selection layer when the algorithm can just do it natively?"

## What You Will Learn

- Utilize algorithms with built-in feature selection (Lasso, Decision Trees, RandomForest)
- Extract `.coef_` to build Feature Importance logic
- Apply `SelectFromModel` meta-transformers

## Prerequisites

- [Wrapper Selection Methods](wrapper-methods.md)

## Step 1: L1 Regularisation (Lasso)

Standard linear regression algorithms will assign a weight (coefficient) to *every single feature* no matter how useless it is. 

**Lasso Regression** (Least Absolute Shrinkage and Selection Operator) utilizes a mathematical penalty called `L1 Regularization`. This penalty forces the coefficients of weak features directly to **0.0**, permanently eliminating them during natural training!

\\[
Loss = \\text{MSE} + \\alpha \\sum_{i=1}^{n} |\\beta_i|
\\]

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

# Load Housing Data
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# ALL Linear Models must be scaled before using L1/L2 penalties!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Standard Linear Regression
lr = LinearRegression()
lr.fit(X_scaled, y)
print("Standard Regression MedInc Coef:", round(lr.coef_[0], 4))

# 2. Lasso Regression (The Embedded Selector)
# Alpha dictates the strength of the penalty. 
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_scaled, y)

print(f"\\nLasso Coefficients (Alpha={lasso.alpha}):")
for feature, coef in zip(data.feature_names, lasso.coef_):
    print(f"{feature}: {round(coef, 4)}")
```

If you ran the script above, you would notice Lasso zeroed out multiple columns natively. The data was filtered without building an external `RFE` pipeline.

## Step 2: Tree-Based Importances

Decision Trees, Random Forests, and Gradient Boosting machines inherently select features by choosing the most optimal splits (Information Gain / Gini Impurity) to build their nodes. We can extract these choices automatically.

```python
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y) # Trees do not require scaling!

# Extract importances
importances = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualizing Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='mako')
plt.title('Random Forest Embedded Feature Importances')
plt.show()
```

!!! tip "Workplace Tip"
    The chart above is the single most common visualization presented to corporate stakeholders during an ML sprint. It answers the fundamental business question: *"What is driving the predictions?"*

## Step 3: Sklearn's `SelectFromModel`

To functionally integrate Embedded methods into a multi-step `Pipeline`, we use `SelectFromModel`. It automatically wraps an estimator and drops columns whose coefficients or importances fall below a set threshold.

```python
from sklearn.feature_selection import SelectFromModel

# We tell SelectFromModel to use the Random Forest logic from Step 2
# and completely drop any feature less important than the median threshold
selector = SelectFromModel(rf, threshold='median')

X_pruned = pd.DataFrame(selector.fit_transform(X, y), 
                        columns=X.columns[selector.get_support()])

print(f"Original shape: {X.shape}")
print(f"Embedded Pruned shape: {X_pruned.shape}")
```

## Summary

Embedded methods are the "Best of Both Worlds". They are faster than Wrapper methods because they only train the model once, and they are more accurate than Filter methods because they learn the interaction between features directly against the prediction target.

## Next Steps

→ [PCA & Dimensionality Reduction](pca-dimensionality-reduction.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S9 | Present findings | Translating mathematical weights into stakeholder impact visualisations |
| S2 | Apply ML techniques | Implementing Lasso regularization and Gini Impurity weighting |
