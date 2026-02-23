# Gradient Boosting Machines

> "If Random Forests rely on the wisdom of an independent crowd, Gradient Boosting relies on an iterative committee, where each new member is hired specifically to fix the mistakes of the previous one."

## What You Will Learn

- Distinguish Boosting from Bagging structurally
- Understand the Gradient Descent optimization of residuals
- Train `GradientBoostingClassifier` and its modern offshoot, `XGBoost`

## Prerequisites

- [Decision Trees & Random Forests](decision-trees.md)

## Step 1: The Concept of Boosting

In a **Random Forest (Bagging)**, 100 deep trees are built independently at the same exact time. They do not communicate. Their final predictions are simply averaged together.

In **Gradient Boosting**, the trees are built *sequentially*. 
1. Tree 1 is built. It is intentionally kept very shallow and weak. It makes predictions.
2. The algorithm calculates the **Residuals** (the errors: Actual - Predicted) for Tree 1.
3. Tree 2 is built. Instead of trying to predict the original target, Tree 2 is trained explicitly to predict the *Residuals* of Tree 1. 
4. The cycle continues. Tree 100 exclusively targets the micro-errors left over by Tree 99.

\\[
F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)
\\]
*(The additive equation: The model at step $m$ adds a newly scaled tree $h_m$ to the existing ensemble $F_{m-1}$.)*

```mermaid
graph LR
    A[Initial Guess] --> B[Tree 1: Fits Data]
    B --> C[Calculate Error 1]
    C --> D[Tree 2: Fits Error 1]
    D --> E[Calculate Error 2]
    E --> F[Tree 3: Fits Error 2]
    F -.-> G[Final Master Prediction]
```

## Step 2: Implementation (Scikit-Learn)

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the GBM
# learning_rate (gamma) controls how much each tree contributes. 
# Lower rate = need more n_estimators = better generalization.
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbm.fit(X_train, y_train)

print(f"Scikit-Learn GBM Accuracy: {gbm.score(X_test, y_test):.4f}")
```

## Step 3: Extreme Gradient Boosting (XGBoost)

While `GradientBoostingClassifier` is excellent, it is computationally slow because it builds trees strictly sequentially. 

**XGBoost** is an external, highly-optimized library that implements the exact same logic but utilizes advanced hardware operations (parallelized node splitting, cache awareness, and regularization penalties) to train astronomically faster on massive datasets.

> [!NOTE]
> `xgboost` is not part of Scikit-Learn. You must `pip install xgboost`.

```python
import xgboost as xgb

# XGBoost can use the Scikit-Learn API format
xgb_clf = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)

xgb_clf.fit(X_train, y_train)

print(f"XGBoost Accuracy: {xgb_clf.score(X_test, y_test):.4f}")
```

## Summary

If you are entering an open Data Science competition for structured tabular data, XGBoost will likely be the winning algorithm. However, because Boosting aggressively chases errors, it is highly prone to Overfitting if left unchecked (unlike Random Forests which resist overfitting naturally).

## Next Steps

→ [Neural Networks (MLP)](neural-networks.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S2 | Apply machine learning algorithms | Deploying iterative sequential ensembles |
| K2 | Architecture principles | Highlighting algorithmic differences (Bagging vs Boosting) |
