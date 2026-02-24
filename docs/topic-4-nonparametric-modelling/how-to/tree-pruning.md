# How to Prune and Regularise Decision Trees

> Preventing a Decision Tree from memorising noise requires constraining its growth — either before training (pre-pruning) or after (post-pruning).

## Pre-Pruning (Constraint-Based)

Set hyperparameters that stop the tree from growing too deep during training:

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits the maximum number of levels in the tree |
| `min_samples_split` | Requires a minimum number of samples to split a node |
| `min_samples_leaf` | Requires a minimum number of samples in each leaf node |
| `max_features` | Limits the number of features considered at each split |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# Pre-pruned tree
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_tr, y_tr)
print(f"Train: {model.score(X_tr, y_tr):.2f}  Test: {model.score(X_te, y_te):.2f}")
```

## Post-Pruning (Cost-Complexity / `ccp_alpha`)

Post-pruning trains a full tree first, then removes branches that contribute less than a threshold (`ccp_alpha`) to overall accuracy. Higher `ccp_alpha` → more aggressive pruning.

```python
import matplotlib.pyplot as plt

# Find the optimal ccp_alpha via cross-validation
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas

from sklearn.model_selection import cross_val_score
import numpy as np

scores = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=42),
                          X_tr, y_tr, cv=5).mean() for a in alphas]

best_alpha = alphas[np.argmax(scores)]
print(f"Best ccp_alpha: {best_alpha:.4f}")

# Train final pruned tree
pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
pruned.fit(X_tr, y_tr)
print(f"Pruned Test Accuracy: {pruned.score(X_te, y_te):.2f}")
```

!!! tip "Workplace Tip"
    Pre-pruning is simpler and faster; post-pruning (`ccp_alpha`) is more principled. In practice, use `GridSearchCV` to tune either set of parameters systematically.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Controlling tree complexity to prevent overfitting |
