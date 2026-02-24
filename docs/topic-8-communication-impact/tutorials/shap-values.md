# SHAP Values

> SHAP (SHapley Additive exPlanations) provides a mathematically rigorous, game-theoretic approach to explaining model predictions — both globally and locally.

## The Concept

SHAP assigns each feature a **Shapley value** — the average marginal contribution of that feature across all possible feature combinations. It answers: *"How much did each feature contribute to pushing this prediction away from the baseline?"*

## Installation

```bash
pip install shap
```

## Implementation

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10,
                            n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te)
```

## Key Visualisations

### Summary Plot (Global)

Shows feature importance and the direction of each feature's effect:

```python
shap.summary_plot(shap_values[1], X_te, feature_names=feature_names)
```

### Waterfall Plot (Local — Single Prediction)

Explains one prediction step by step:

```python
shap.plots.waterfall(explainer(X_te)[0])
```

### Bar Plot (Global Importance)

Simple bar chart of mean absolute SHAP values:

```python
shap.plots.bar(explainer(X_te))
```

## SHAP vs LIME

| Aspect | SHAP | LIME |
|--------|------|------|
| Theory | Game-theoretic (exact) | Local linear approximation |
| Consistency | Guaranteed consistent | No consistency guarantees |
| Speed | Slower (especially KernelSHAP) | Faster |
| Global view | Yes (summary plot) | No (local only) |
| Best for | Thorough analysis and reports | Quick local explanations |

## Tree-Specific Speedup

For tree-based models, `TreeExplainer` computes exact SHAP values in polynomial time — much faster than the model-agnostic `KernelExplainer`.

!!! tip "Workplace Tip"
    SHAP is the gold standard for model explainability. Include a SHAP summary plot in every ML report — it shows stakeholders which features matter and how they influence predictions in a single chart.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Providing rigorous model explanations using SHAP |
