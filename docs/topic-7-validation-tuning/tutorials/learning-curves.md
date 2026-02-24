# Learning Curves

> Learning curves plot model performance against training set size. They diagnose whether your model suffers from high bias (underfitting) or high variance (overfitting).

## How to Read Them

| Pattern | Training Score | Validation Score | Diagnosis | Fix |
|---------|---------------|-----------------|-----------|-----|
| Both low | Low | Low | High bias (underfitting) | Use a more complex model or add features |
| Big gap | High | Low | High variance (overfitting) | Get more data, regularise, or simplify the model |
| Both high, converging | High | High (close) | Good fit | You're done |

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_mean, "o-", label="Training Score", color="blue")
plt.plot(train_sizes, val_mean, "o-", label="Validation Score", color="orange")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

## Interpreting Results

- **Converging curves with a small gap:** Your model generalises well. More data is unlikely to help.
- **Large gap that shrinks with more data:** High variance — the model overfits but more data will help.
- **Both curves plateau at a low score:** High bias — a more powerful model or better features are needed.

!!! tip "Workplace Tip"
    Always plot a learning curve before asking for more data. If the curves have already converged, collecting more data will not improve performance — you need a better model or better features.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Diagnosing model performance with learning curves |
