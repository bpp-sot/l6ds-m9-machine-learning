# How to Prevent Overfitting (Validation Strategy)

> Overfitting occurs when your model learns training noise instead of general patterns. Proper validation is your primary defence.

## Signs of Overfitting

| Metric | Training Set | Test Set | Diagnosis |
|--------|-------------|----------|-----------|
| Accuracy | 0.99 | 0.72 | Overfitting — large gap between train and test |
| Accuracy | 0.85 | 0.83 | Good generalisation — small gap |

## Prevention Strategies

### 1. Cross-Validation

Never evaluate on a single train/test split. Use k-Fold CV to get a robust estimate:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 2. Regularisation

Add penalties to model complexity (see [Regularisation](../../topic-3-predictive-modelling/explanation/regularisation.md)):

```python
from sklearn.linear_model import LogisticRegression

# C controls inverse regularisation strength — smaller C = more regularisation
lr = LogisticRegression(C=0.1, penalty="l2", max_iter=1000)
```

### 3. Reduce Model Complexity

Constrain hyperparameters to prevent the model from memorising data:

```python
from sklearn.tree import DecisionTreeClassifier

# Limit tree growth
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
```

### 4. Early Stopping

For iterative algorithms, stop training when validation error starts increasing:

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=10,
    tol=0.001
)
```

### 5. More Data

Sometimes the simplest fix is more training data. Overfitting is fundamentally a problem of having too many parameters relative to the number of observations.

!!! warning "Common Pitfall"
    Tuning hyperparameters on the test set causes **information leakage**. Always use a separate validation set (or nested CV) for tuning, and reserve the test set for the final evaluation only.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Systematic strategies to prevent overfitting |
