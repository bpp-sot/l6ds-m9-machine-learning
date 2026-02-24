# Train/Test Split

> The most fundamental validation step: hold out a portion of your data that the model never sees during training, then evaluate on it.

## Why Split?

If you evaluate a model on the same data it trained on, you measure how well it **memorises**, not how well it **generalises**. The test set simulates unseen, real-world data.

## Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 80% train, 20% test
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y           # Preserve class proportions
)

model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)
print(f"Test Accuracy: {model.score(X_te, y_te):.4f}")
```

## Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `test_size` | Fraction of data for testing (default 0.25) |
| `random_state` | Seed for reproducibility |
| `stratify=y` | Preserve class distribution — **always use for classification** |
| `shuffle=True` | Shuffle before splitting (default) — **set False for time series** |

## Train / Validation / Test Split

For hyperparameter tuning, you need **three** sets:

```python
# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Second split: train and validation from the remaining data
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

| Set | Purpose |
|-----|---------|
| **Train** | Fit the model |
| **Validation** | Tune hyperparameters and select the best model |
| **Test** | Final, unbiased evaluation — touch it **once** |

!!! warning "Common Pitfall"
    Never tune hyperparameters on the test set. If you do, the test score is no longer an unbiased estimate of real-world performance. Use cross-validation or a separate validation set for tuning.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
