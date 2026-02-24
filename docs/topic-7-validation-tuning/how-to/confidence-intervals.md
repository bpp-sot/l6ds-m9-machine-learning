# How to Compute Confidence Intervals for Model Performance

> A single accuracy number is meaningless without a confidence interval. Report the range in which the true performance likely falls.

## Why Confidence Intervals?

A model scoring 0.85 accuracy on one test set might score 0.82 or 0.88 on a different split. Confidence intervals quantify this uncertainty.

## Method: Bootstrap Resampling

Repeatedly resample predictions with replacement and compute the metric on each sample to build a distribution.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
preds = model.predict(X_te)

# Bootstrap confidence interval
rng = np.random.default_rng(42)
n_bootstrap = 1000
scores = []

for _ in range(n_bootstrap):
    idx = rng.choice(len(y_te), size=len(y_te), replace=True)
    scores.append(accuracy_score(y_te.iloc[idx] if hasattr(y_te, 'iloc') else y_te[idx],
                                  preds[idx]))

lower = np.percentile(scores, 2.5)
upper = np.percentile(scores, 97.5)
print(f"Accuracy: {np.mean(scores):.4f}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
```

## Method: Cross-Validation Interval

A simpler (less rigorous) approach uses the standard deviation across CV folds:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=10, scoring="accuracy")
mean = scores.mean()
ci = 1.96 * scores.std()  # Approximate 95% CI
print(f"Accuracy: {mean:.4f} ± {ci:.4f}")
print(f"95% CI: [{mean - ci:.4f}, {mean + ci:.4f}]")
```

!!! tip "Workplace Tip"
    Always report model performance as a range, not a point estimate. Stakeholders and EPA assessors will be more convinced by "accuracy of 0.85 ± 0.03" than "accuracy of 0.85".

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Quantifying uncertainty in model performance estimates |
