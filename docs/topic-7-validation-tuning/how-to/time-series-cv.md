# Time Series Cross-Validation

> Standard k-Fold CV shuffles data randomly, which is invalid for time series. Use **expanding window** or **sliding window** CV instead.

## Why Standard CV Fails for Time Series

Standard k-Fold randomly assigns observations to folds, allowing the model to train on future data and test on past data. This leaks temporal information and produces artificially high scores.

## TimeSeriesSplit

Scikit-learn's `TimeSeriesSplit` creates expanding training windows while always testing on the next chronological block:

```
Fold 1: [Train] | [Test]
Fold 2: [Train    Train] | [Test]
Fold 3: [Train    Train    Train] | [Test]
```

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Simulate 200 time-ordered observations
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = X[:, 0] * 2 + rng.normal(size=200)

tscv = TimeSeriesSplit(n_splits=5)

scores = cross_val_score(
    RandomForestRegressor(random_state=42),
    X, y,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)
print(f"Time Series CV MAE: {-scores.mean():.3f} ± {scores.std():.3f}")
```

## Visualising the Splits

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    ax.barh(i, len(train_idx), left=train_idx[0], height=0.4, color="steelblue", label="Train" if i == 0 else "")
    ax.barh(i, len(test_idx), left=test_idx[0], height=0.4, color="coral", label="Test" if i == 0 else "")

ax.set_xlabel("Sample Index")
ax.set_ylabel("CV Fold")
ax.set_title("TimeSeriesSplit — Expanding Window")
ax.legend()
plt.tight_layout()
plt.show()
```

!!! tip "Workplace Tip"
    For very long time series, use a `gap` parameter between train and test to simulate realistic prediction horizons (e.g., predicting 7 days ahead, not the next day).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Applying temporally valid cross-validation for time series |
