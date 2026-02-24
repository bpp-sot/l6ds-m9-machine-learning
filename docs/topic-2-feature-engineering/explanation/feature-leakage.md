# Feature Leakage

> Target Leakage destroys algorithmic validity by revealing the answer to the model during training.

## The Concept

Target Leakage (also called Data Leakage) occurs when you engineer a feature that contains information which will **not be available at prediction time** in the real world.

Your model achieves near-perfect accuracy during training and validation, but collapses entirely when deployed to production — because the leaked signal no longer exists.

## Example: The Churn Prediction Disaster

Imagine building a model to predict whether a user will cancel their subscription (churn) next month.

During feature engineering, you create a column: `has_called_cancellation_hotline_last_30_days`.

**The Leakage Flaw:**

Your algorithm discovers that `has_called_cancellation_hotline` has a 99.9% correlation with the target variable `churned`. It assigns almost all predictive weight to this single feature, achieving 99% accuracy in cross-validation.

**The Production Failure:**

When you deploy the model to predict churn for *next* month, that column does not yet exist — you cannot know today whether a customer will call the cancellation hotline over the coming 30 days. The model's star feature is empty, and predictions collapse to random guessing.

## How to Detect Leakage

1. **Suspiciously high accuracy.** If your model achieves > 95% accuracy with minimal tuning, investigate which features are driving it.
2. **Single-feature dominance.** If one feature has overwhelming importance, check whether it is temporally valid.
3. **Timeline audit.** For every feature, ask: "Would I physically have this value *before* the event I am predicting?"

## The Solution

You must enforce a strict **temporal cutoff**: only include features derived from data available *before* the prediction point.

```python
# Correct: features derived from data BEFORE the prediction window
df["calls_previous_quarter"] = df.groupby("user_id")["call_date"].transform(
    lambda x: x.shift(1).rolling("90D").count()
)
```

!!! warning "Common Pitfall"
    Fitting a scaler or encoder on the full dataset (including test data) before splitting is also a form of leakage. Always fit transformers on the training set only, then apply `.transform()` to the test set.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-----------------------|
| K5 | Machine Learning workflows | Avoiding temporal leakage through disciplined feature engineering |
| B2 | Logical and analytical approach | Auditing feature validity against real-world deployment constraints |
