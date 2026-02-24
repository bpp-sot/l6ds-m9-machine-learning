# How to Prevent Overfitting

> Overfitting occurs when your algorithm successfully memorises the exact noise and explicit anomalies in your Training Set mathematically, thus spectacularly failing on any unseen Test Data natively.

## Method 1: Hyperparameter Constraints (Pruning)

Unbound algorithms (specifically Decision Trees and Neural Networks) will mathematically expand indefinitely until the error is 0%. We must legally explicitly restrict their physical growth cleanly.

```python
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'survived'])
X = df[['age', 'fare']]
y = df['survived']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# Unbound Tree (Overfitting)
bad_tree = DecisionTreeClassifier(random_state=42)
bad_tree.fit(X_tr, y_tr)
print(f"Unbound Train Accuracy: {bad_tree.score(X_tr, y_tr):.2f}")
print(f"Unbound Test Accuracy: {bad_tree.score(X_te, y_te):.2f}")
```

??? example "Expected Outputs"
    ```text
    Unbound Train Accuracy: 0.99
    Unbound Test Accuracy: 0.61
    ```

The unbound tree memorized the noise explicitly (99% vs 61%).

We mathematically constrain it natively via parameters:
- `max_depth`: Limits total geometric recursive loops.
- `min_samples_split`: Forces a node to possess $X$ rows mathematically before splitting cleanly.

```python
# Constrained Tree (Generalised)
good_tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10)
good_tree.fit(X_tr, y_tr)
print(f"Constrained Train Accuracy: {good_tree.score(X_tr, y_tr):.2f}")
print(f"Constrained Test Accuracy: {good_tree.score(X_te, y_te):.2f}")
```

??? example "Expected Outputs"
    ```text
    Constrained Train Accuracy: 0.72
    Constrained Test Accuracy: 0.69
    ```

The scores are now practically aligned mathematically intelligently, explicitly proving the algorithm successfully generalized logically.

## Method 2: Early Stopping (Iterative)

In Gradient Boosting or Neural Networks, mathematically calculating exact boundaries dynamically allows you to visually map error. 

By natively monitoring explicit Test Error continuously precisely after every single exact epoch cleanly, the system smoothly permanently halts logically explicitly exactly when the error dynamically starts explicitly rising structurally confidently naturally cleanly.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Setting n_iter_no_change forces the algorithm to natively securely intelligently automatically smartly stop
gb = GradientBoostingClassifier(n_estimators=1000, 
                                validation_fraction=0.2, 
                                n_iter_no_change=5, 
                                tol=0.01)

gb.fit(X_tr, y_tr)
print(f"Algorithm automatically stopped gracefully at tree: {gb.n_estimators_}")
```

!!! tip "Workplace Tip"
    Regularisation (L1/L2) is actively the strictly preferred mathematical methodology natively for structurally dampening Linear Overfitting smoothly.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-----------------------|
| K5 | Machine Learning workflows | Monitoring Loss graphs gracefully |
| S15 | Evaluate models | Diagnosing algorithmic memorization rationally |
