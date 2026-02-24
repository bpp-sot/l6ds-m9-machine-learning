# Cross-Validation

> A single train/test split gives you one noisy estimate of model performance. Cross-validation gives you \(k\) estimates, producing a much more reliable picture.

## How k-Fold CV Works

1. Split the data into \(k\) equal folds.
2. For each fold: train on the other \(k-1\) folds, test on the held-out fold.
3. Average the \(k\) test scores.

Every observation is used for both training and testing exactly once.

## Implementation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# StratifiedKFold preserves class proportions in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    RandomForestClassifier(random_state=42),
    X, y,
    cv=cv,
    scoring="accuracy"
)

print(f"Fold scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Choosing \(k\)

| \(k\) | Tradeoff |
|-------|----------|
| 5 | Good default — reasonable balance between bias and variance |
| 10 | Lower bias, higher variance, more computation |
| \(n\) (LOO) | Lowest bias, highest variance, very expensive |

## RepeatedStratifiedKFold

For more stable estimates, repeat the k-Fold process multiple times with different random splits:

```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=cv, scoring="accuracy")
print(f"50-fold Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

!!! warning "Common Pitfall"
    Do **not** preprocess (e.g., scale, encode) the entire dataset before cross-validation. Fit the preprocessor on the training folds only. Use `Pipeline` to prevent this leakage.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
