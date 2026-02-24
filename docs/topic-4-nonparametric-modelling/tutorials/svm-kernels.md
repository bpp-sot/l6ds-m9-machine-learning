# SVM and Kernel Methods

> The true power of Support Vector Machines lies not in linear separation, but in the **kernel trick** — projecting data into a higher-dimensional space where a linear boundary becomes possible.

## How SVMs Work

An SVM finds the hyperplane that maximises the **margin** — the distance between the decision boundary and the nearest data points from each class (the "support vectors").

- **Linear SVM:** Draws a straight line (or hyperplane) between classes. Works only when classes are linearly separable.
- **Kernel SVM:** Implicitly maps data into a higher-dimensional space where a linear separator exists, without ever computing the transformation explicitly.

## The Kernel Trick

Instead of transforming your data into a high-dimensional space (which would be computationally expensive), the kernel trick computes the **dot product** in that space directly. This gives you the power of non-linear boundaries at a fraction of the cost.

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# Compare linear vs RBF kernel
for kernel in ["linear", "rbf"]:
    svc = SVC(kernel=kernel, random_state=42)
    svc.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, svc.predict(X_te))
    print(f"{kernel:>6s} kernel accuracy: {acc:.2f}")
```

## Visualising the Decision Boundary

```python
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
)

svc_rbf = SVC(kernel="rbf", random_state=42).fit(X_tr, y_tr)
Z = svc_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.title("SVM with RBF Kernel — Non-Linear Boundary")
plt.tight_layout()
plt.show()
```

## Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `C` | Regularisation strength — higher values fit training data more tightly |
| `gamma` | RBF reach — higher values create tighter, more localised boundaries |
| `kernel` | `"linear"`, `"rbf"`, `"poly"`, `"sigmoid"` |

!!! warning "Common Pitfall"
    SVMs are sensitive to feature scaling. Always standardise your features with `StandardScaler` before fitting.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Understanding kernel methods and their role in non-linear classification |
