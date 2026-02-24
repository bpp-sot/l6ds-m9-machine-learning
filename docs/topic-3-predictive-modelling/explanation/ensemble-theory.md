# Ensemble Theory Explained

> "The wisdom of the crowd" — aggregating many weak models into one strong model is almost always superior to relying on a single algorithm.

## The Core Idea

A single Decision Tree is noisy and unstable: small changes in the training data can produce a completely different tree. However, if you train 100 slightly different trees and let them **vote**, individual errors cancel out and the collective prediction stabilises.

This principle underpins all ensemble methods.

## Bagging (Bootstrap Aggregating)

Each base model is trained on a **random bootstrap sample** (sampling with replacement) of the original dataset. Predictions are combined by majority vote (classification) or averaging (regression).

* **Key algorithm:** `RandomForestClassifier` / `RandomForestRegressor`
* **Effect:** Reduces variance without increasing bias.

```python
from sklearn.ensemble import RandomForestClassifier

# 100 independent trees, each trained on a bootstrapped sample
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

## Boosting (Sequential Correction)

Each base model is trained **sequentially**, with each new model focusing on the mistakes of its predecessor. This progressively reduces bias.

* **Key algorithms:** `GradientBoostingClassifier`, `XGBClassifier`, `LGBMClassifier`
* **Effect:** Reduces bias, but can overfit if not regularised (use `learning_rate`, `max_depth`, `n_estimators`).

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
```

## Stacking

Multiple heterogeneous models (e.g., a Logistic Regression, a Random Forest, and an SVM) each make predictions. A **meta-learner** then combines those predictions into a final output.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier()), ("svm", SVC())],
    final_estimator=LogisticRegression()
)
```

!!! tip "Workplace Tip"
    In production, Gradient Boosting frameworks (XGBoost, LightGBM) dominate tabular data competitions and real-world deployments because they combine high accuracy with built-in regularisation controls.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
