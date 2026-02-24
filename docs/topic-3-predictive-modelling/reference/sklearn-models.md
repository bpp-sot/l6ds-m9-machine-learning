# Scikit-Learn Model API Structure

> The genius of scikit-learn is that every algorithm — from Linear Regression to Neural Networks — follows the exact same 4-step API.

## The Core 4 Commands

Regardless of whether you deploy a `LinearRegression` or a `MLPClassifier`, the code structure remains identical.

### 1. Instantiate

Create the model object in memory with your chosen hyperparameters.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### 2. Fit

Pass your training data so the algorithm learns patterns.

```python
model.fit(X_train, y_train)
```

### 3. Predict

Generate predictions on new, unseen data.

```python
predictions = model.predict(X_test)
```

### 4. Score

Extract the default evaluation metric (accuracy for classifiers, \(R^2\) for regressors).

```python
accuracy = model.score(X_test, y_test)
```

## Special Extension Methods

| Method | Available On | Returns |
|--------|-------------|---------|
| `predict_proba()` | Classification models | Probability per class, e.g., `[0.12, 0.88]` |
| `decision_function()` | SVM, Linear models | Raw distance from the decision boundary |
| `transform()` | Preprocessors, PCA | Transformed feature matrix |
| `fit_transform()` | Preprocessors, PCA | Fit + transform in a single call |

## Pipeline Integration

You can chain preprocessing and modelling into a single `Pipeline` object that still follows the same `fit` / `predict` / `score` API:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipe.fit(X_train, y_train)
print(f"Accuracy: {pipe.score(X_test, y_test):.2f}")
```

!!! tip "Workplace Tip"
    Always wrap your workflow in a `Pipeline`. It prevents data leakage (the scaler is fitted only on training data) and makes your code reproducible and deployable.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S13 | Apply ML algorithms | Using the consistent scikit-learn API across all model types |
