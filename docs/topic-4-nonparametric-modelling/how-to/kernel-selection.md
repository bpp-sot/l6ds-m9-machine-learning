# How to Select an SVM Kernel

> A linear kernel is fast but limited to linearly separable data. The RBF kernel is the default for complex, non-linear boundaries.

## The Strategy

Your kernel choice depends on the geometry of your data:

| Kernel | When to Use | Strengths |
|--------|-------------|-----------|
| `linear` | Data is linearly separable; high-dimensional sparse data (e.g., text) | Fast to train, interpretable coefficients |
| `rbf` (default) | Non-linear relationships; moderate feature count | Flexible, captures complex boundaries |
| `poly` | Known polynomial relationship between features | Tuneable degree parameter |

## Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Compare kernels
for kernel in ["linear", "rbf", "poly"]:
    svc = SVC(kernel=kernel, random_state=42)
    score = cross_val_score(svc, X, y, cv=5, scoring="accuracy").mean()
    print(f"{kernel:>6s}: {score:.3f}")
```

## Tuning RBF Hyperparameters

The RBF kernel has two critical hyperparameters:

- **`C`** (regularisation): Controls the tradeoff between a smooth boundary and correctly classifying training points. High `C` → tighter fit (risk of overfitting).
- **`gamma`**: Controls the "reach" of each training point. High `gamma` → each point only influences its immediate neighbours (risk of overfitting).

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.3f}")
```

!!! tip "Workplace Tip"
    Start with `SVC(kernel='rbf')` using default `C=1` and `gamma='scale'`. Only switch to a linear kernel if you have very high-dimensional data (e.g., thousands of text features) where RBF becomes computationally expensive.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced ML techniques | Tree-based models, ensemble methods, KNN, SVM |
| K4.4 | Trade-offs in selecting algorithms | Comparing parametric vs non-parametric approaches |
| S4 | ML and optimisation | Hyperparameter tuning, ensemble construction, model selection |
| B1 | Curiosity and creativity | Exploring when non-parametric methods outperform parametric ones |
| B5 | Integrity in presenting conclusions | Avoiding overfitting; honest reporting of generalisation performance |
