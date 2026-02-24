# Feature Importance Visualisation

> Show stakeholders which features drive your model's predictions using built-in feature importance scores.

## Tree-Based Feature Importance

Tree-based models (Random Forest, Gradient Boosting, XGBoost) compute feature importance natively based on how much each feature reduces impurity across all trees.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10,
                            n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extract and sort importances
importance = pd.Series(model.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importance.plot.barh(color="steelblue")
plt.xlabel("Importance (Mean Decrease in Impurity)")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.show()
```

## Permutation Importance

A model-agnostic approach: shuffle each feature and measure how much the score drops. Larger drops indicate more important features.

```python
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)
model.fit(X_tr, y_tr)

result = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=42)

perm_imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
perm_imp.plot.barh(color="coral")
plt.xlabel("Mean Accuracy Decrease")
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()
```

## Tree vs Permutation Importance

| Method | Pros | Cons |
|--------|------|------|
| Tree-based (MDI) | Fast, built-in | Biased towards high-cardinality features |
| Permutation | Model-agnostic, unbiased | Slower, affected by correlated features |

!!! tip "Workplace Tip"
    Use tree-based importance for initial exploration and permutation importance for your final report. Stakeholders respond well to simple bar charts showing "which features matter most."

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Communicating model drivers to stakeholders |
