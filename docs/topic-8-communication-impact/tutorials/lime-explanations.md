# LIME Explanations

> LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by fitting a simple, interpretable model around a single data point.

## The Concept

LIME answers: *"Why did the model predict X for this specific observation?"*

It works by:

1. Perturbing the input — creating many slightly modified copies of the observation.
2. Getting predictions from the black-box model for each perturbation.
3. Fitting a simple linear model (weighted by proximity) to approximate the black-box locally.
4. Reporting which features pushed the prediction up or down.

## Installation

```bash
pip install lime
```

## Implementation — Tabular Data

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

X, y = make_classification(n_samples=1000, n_features=10,
                            n_informative=5, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

# Create the LIME explainer
explainer = LimeTabularExplainer(
    X_tr,
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    mode="classification"
)

# Explain a single prediction
idx = 0  # First test observation
explanation = explainer.explain_instance(X_te[idx], model.predict_proba, num_features=5)

# Show in notebook
explanation.show_in_notebook()

# Or get as a list
print("Top contributing features:")
for feat, weight in explanation.as_list():
    print(f"  {feat}: {weight:+.4f}")
```

## LIME vs Global Feature Importance

| Aspect | Global Importance | LIME |
|--------|------------------|------|
| Scope | Entire dataset | Single prediction |
| Question | "Which features matter overall?" | "Why this prediction?" |
| Use case | Model summary | Explaining individual decisions to stakeholders |

!!! tip "Workplace Tip"
    LIME is invaluable when you need to justify a specific model decision — for example, explaining to a customer why their loan application was declined. Pair it with SHAP for a complete explainability toolkit.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S5 | Deployment, value assessment, and ROI | Translating model performance into business impact |
| S6 | Communicate through storytelling and visualisation | Presenting ML results to non-technical stakeholders |
| B4 | Consideration of organisational goals | Framing technical results in terms of business objectives |
| B1 | Inquisitive approach | Exploring creative ways to explain model behaviour |
