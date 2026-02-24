# How to Choose the Right \(k\) Value

> The \(k\) hyperparameter in k-Nearest Neighbours dictates whether your algorithm overfits (low \(k\)) or underfits (high \(k\)). Choosing the right value is critical.

## The Tradeoff

| \(k\) Value | Behaviour | Risk |
|------------|-----------|------|
| Low (e.g., 1–3) | Highly sensitive to individual data points | Overfitting — noisy, jagged decision boundaries |
| High (e.g., 50+) | Over-smoothed, ignores local patterns | Underfitting — the model defaults to majority class |

## The Elbow Method

Sweep a range of \(k\) values using cross-validation and plot accuracy against \(k\). Choose the value where accuracy plateaus — the "elbow" of the curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

k_range = range(1, 31)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
          for k in k_range]

plt.figure(figsize=(8, 4))
plt.plot(k_range, scores, marker="o")
plt.xlabel("k")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Elbow Plot for k Selection")
plt.tight_layout()
plt.show()
```

## Using GridSearchCV

For a more automated approach, let scikit-learn select the best \(k\) for you:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
    cv=5,
    scoring="accuracy"
)
grid.fit(X, y)
print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"Best CV Accuracy: {grid.best_score_:.3f}")
```

!!! tip "Workplace Tip"
    Always use an **odd** value for \(k\) in binary classification to avoid tied votes.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced ML techniques | Tree-based models, ensemble methods, KNN, SVM |
| K4.4 | Trade-offs in selecting algorithms | Comparing parametric vs non-parametric approaches |
| S4 | ML and optimisation | Hyperparameter tuning, ensemble construction, model selection |
| B1 | Curiosity and creativity | Exploring when non-parametric methods outperform parametric ones |
| B5 | Integrity in presenting conclusions | Avoiding overfitting; honest reporting of generalisation performance |
