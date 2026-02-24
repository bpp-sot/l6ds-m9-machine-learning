# The Bias-Variance Tradeoff in Modelling

> In predictive modelling, Bias and Variance represent a mathematical scale. You must balance the algorithm to sit in the middle.

## High Bias (Underfitting)

An algorithm with high bias (e.g., Linear Regression on a non-linear problem) makes overly simplistic assumptions and ignores genuine complexity in the data.

* **Symptom:** The model scores poorly on *both* the Training data and the Test data.
* **Analogy:** Studying only Chapter 1 for a ten-chapter final exam — you lack the depth to answer most questions.

## High Variance (Overfitting)

An algorithm with high variance (e.g., an unbound Decision Tree) memorises every quirk and noise artefact in the training set, then fails on anything new.

* **Symptom:** Scores brilliantly on Training, but drastically poorly on Validation/Test.
* **Analogy:** Memorising the exact wording of past exam papers rather than learning the underlying principles — any rephrased question defeats you.

## The Sweet Spot

The goal of model tuning is to find the point where the combined error from bias and variance is minimised. You achieve this by:

1. **Increasing model complexity** (reducing bias) — e.g., adding polynomial features, increasing tree depth.
2. **Applying regularisation** (reducing variance) — e.g., L1/L2 penalties, `max_depth` limits, early stopping.
3. **Using cross-validation** to measure generalisation performance at each complexity level.

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Sweep max_depth to observe bias-variance tradeoff
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name="max_depth", param_range=range(1, 20),
    cv=5, scoring="accuracy"
)
```

When you plot these curves, the gap between training and validation scores reveals the variance; a low overall score reveals bias.

!!! info "Assessment Connection"
    Explicitly discussing the bias-variance tradeoff in your EPA demonstrates you understand *why* you tuned hyperparameters, not just *how*.

## KSB Mapping

| KSB | Description | How This Explanation Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Understanding the fundamental tradeoff governing model selection |
