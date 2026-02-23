# Decision Trees & Random Forests

> "A decision tree is essentially a highly-optimized flowchart created by a machine rather than a manager."

## What You Will Learn

- Build mathematical branching logic using Gini Impurity or Information Gain
- Construct a single `DecisionTreeClassifier`
- Overcome Tree Overfitting using Bootstrap Aggregation (`RandomForestClassifier`)

## Prerequisites

- [Pipelines in Data Prep](../../topic-1-data-preparation/tutorials/pipelines.md)

## Step 1: Anatomy of a Split

Unlike SVMs or Linear models, Decision Trees do not use equations to draw lines through space. They slice the dataset logically using IF/ELSE conditionals orthogonal (perpendicular) to the axes.

To decide *which* feature to split on at the structural "Root", the algorithm calculates **Gini Impurity** or **Entropy**.

\\[
Gini = 1 - \\sum_{i=1}^{C} (p_i)^2
\\]
*(A mathematically "pure" node where every sample belongs to the exact same class has a Gini score of 0.0).*

The algorithm tests thousands of feature boundaries (e.g. `Age < 35`, `Age < 36`) and permanently chooses the split that creates the purest child nodes (Maximizing Information Gain).

## Step 2: The Base Decision Tree

Because Trees use boolean logic rather than distance algorithms, they **DO NOT require feature scaling** and natively ignore extreme outliers.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load Data
iris = load_iris()
X, y = iris.data, iris.target

# 1. Instantiate the Tree
# Limiting depth prevents the tree from creating 1 leaf per sample (Overfitting)
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# 2. Fit the Model
tree_clf.fit(X, y)

# 3. Predict Evaluation
print(f"Accuracy: {tree_clf.score(X, y):.4f}")

# Visualizing the Flowchart
plt.figure(figsize=(14, 8))
plot_tree(tree_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

## Step 3: The Fatal Flaw

A single Decision Tree will naturally grow until every single Leaf has 1 sample (Gini = 0.0). This means the tree permanently memorizes the training data noise and functionally achieves 100% variance. It is completely useless for predicting novel data unless you strictly prune it via `max_depth`.

## Step 4: Ensembles (The Random Forest Solution)

If one tree is prone to errors, we group them into a "Forest". 

A **Random Forest** is an ensemble algorithm employing a technique called **Bagging** (Bootstrap Aggregation):
1. **Bootstrapping**: It builds 100 different Trees. Each tree is trained on a random, distinct subset of the rows (sampled with replacement).
2. **Feature Randomness**: At every split node, the tree is only allowed to look at a random subset of the columns (e.g., `<sqrt(features)>`). 
3. **Aggregation**: When predicting, all 100 trees vote. The majority wins.

Because every tree is structurally unique and conceptually narrow, when they aggregate their votes, the underlying noise cancels out, creating a highly resilient, low-variance model.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate the Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

rf_clf.fit(X_train, y_train)

print(f"Random Forest Accuracy: {rf_clf.score(X_test, y_test):.4f}")
```

!!! success "Assessment Checklist"
    If you use a Random Forest in your presentation, you must be prepared to articulate the difference between a `Single Weak Learner` (The Tree) and the `Aggregated Meta-Estimator` (The Forest). Do not confuse them conceptually.

## Summary

In modern Data Science, Ensembled Trees execute tabular prediction faster, more robustly, and require less preprocessing than any other algorithm class. Random Forests are the workhorses of the ML industry.

## Next Steps

→ [Gradient Boosting Machines](gradient-boosting.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| K2 | Architecture principles | Articulating boolean splitting vs geometric plotting |
| S2 | Apply algorithms | Utilizing Bagging to solve high variance structures |
