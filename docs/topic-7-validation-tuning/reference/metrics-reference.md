# Metrics Reference

> A complete reference of evaluation metrics for classification and regression.

## Classification Metrics

| Metric | Formula / Description | When to Use |
|--------|----------------------|-------------|
| **Accuracy** | Correct predictions / Total predictions | Balanced classes only |
| **Precision** | TP / (TP + FP) | When false positives are costly (e.g., spam detection) |
| **Recall** | TP / (TP + FN) | When false negatives are costly (e.g., disease detection) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | When you need a balance between precision and recall |
| **ROC AUC** | Area under the ROC curve | Comparing classifiers across all thresholds |
| **Log Loss** | -mean(y·log(p) + (1-y)·log(1-p)) | When you care about probability calibration |

## Regression Metrics

| Metric | Formula / Description | When to Use |
|--------|----------------------|-------------|
| **MAE** | Mean of \|actual - predicted\| | Robust to outliers; easy to interpret |
| **MSE** | Mean of (actual - predicted)² | Penalises large errors heavily |
| **RMSE** | √MSE | Same units as the target variable |
| **R²** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0–1 for good models) |
| **MAPE** | Mean of \|error / actual\| × 100 | Percentage error — useful for comparing across scales |

## Quick Implementation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

# Classification
y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred_cls = [1, 0, 1, 0, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true_cls, y_pred_cls):.3f}")
print(f"Precision: {precision_score(y_true_cls, y_pred_cls):.3f}")
print(f"Recall:    {recall_score(y_true_cls, y_pred_cls):.3f}")
print(f"F1:        {f1_score(y_true_cls, y_pred_cls):.3f}")

# Regression
y_true_reg = np.array([3.0, 5.0, 2.5, 7.0])
y_pred_reg = np.array([2.8, 5.2, 2.1, 6.5])

print(f"\nMAE:  {mean_absolute_error(y_true_reg, y_pred_reg):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)):.3f}")
print(f"R²:   {r2_score(y_true_reg, y_pred_reg):.3f}")
```

!!! warning "Common Pitfall"
    **Never use accuracy on imbalanced data.** A model that always predicts the majority class achieves high accuracy but is useless. Use F1, Precision/Recall, or ROC AUC instead.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Selecting appropriate evaluation metrics |
