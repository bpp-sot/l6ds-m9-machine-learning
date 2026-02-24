# How to Prevent Overfitting

> Overfitting occurs when your algorithm memorises the noise and anomalies in your training set, then fails on any unseen test data.

## Method 1: Hyperparameter Constraints (Pruning)

Unbounded algorithms (Decision Trees, Neural Networks) will expand indefinitely until training error hits zero. You must restrict their growth.

```python
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = sns.load_dataset("titanic").dropna(subset=["age", "fare", "survived"])
X = df[["age", "fare"]]
y = df["survived"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# Unbound Tree (Overfitting)
bad_tree = DecisionTreeClassifier(random_state=42)
bad_tree.fit(X_tr, y_tr)
print(f"Unbound Train Accuracy: {bad_tree.score(X_tr, y_tr):.2f}")
print(f"Unbound Test Accuracy:  {bad_tree.score(X_te, y_te):.2f}")
```

??? example "Expected Output"
    ```text
    Unbound Train Accuracy: 0.99
    Unbound Test Accuracy:  0.61
    ```

The unbound tree memorised the training noise (99% vs 61% — a huge gap).

Now constrain it using `max_depth` (limits tree depth) and `min_samples_split` (requires a minimum number of samples to split a node):

```python
good_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
good_tree.fit(X_tr, y_tr)
print(f"Constrained Train Accuracy: {good_tree.score(X_tr, y_tr):.2f}")
print(f"Constrained Test Accuracy:  {good_tree.score(X_te, y_te):.2f}")
```

??? example "Expected Output"
    ```text
    Constrained Train Accuracy: 0.72
    Constrained Test Accuracy:  0.69
    ```

The scores are now closely aligned, proving the algorithm generalises rather than memorises.

## Method 2: Early Stopping (Iterative Models)

For Gradient Boosting or Neural Networks, you can monitor validation error during training and halt automatically when performance stops improving.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=5,
    tol=0.01,
    random_state=42
)
gb.fit(X_tr, y_tr)
print(f"Algorithm stopped at iteration: {gb.n_estimators_}")
```

## Method 3: Regularisation

For linear models, apply L1 (Lasso) or L2 (Ridge) penalties to shrink or eliminate coefficients. See the [Regularisation Explained](../explanation/regularisation.md) page for details.

!!! tip "Workplace Tip"
    Regularisation is the preferred method for preventing overfitting in linear models; hyperparameter constraints (pruning) are the equivalent for tree-based models.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-----------------------|
| K5 | Machine Learning workflows | Monitoring training vs validation loss to detect overfitting |
| S15 | Evaluate model performance | Diagnosing and correcting algorithmic memorisation |
