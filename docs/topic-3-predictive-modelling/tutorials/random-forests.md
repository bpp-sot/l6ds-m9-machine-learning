# Random Forests

> A single Decision Tree is weak and prone to overfitting. A forest of 100 independent decision trees voting together is incredibly robust.

## What You Will Learn
- Define Ensemble Learning conceptually 
- Explain Bagging (Bootstrap Aggregation)
- Deploy `RandomForestClassifier` utilizing Scikit-Learn

## Prerequisites
- Completed the *Decision Trees* tutorial
- Basic understanding of classification voting schemas

## Step 1: Wisdom of the Crowds (Ensemble)

If you ask one student to guess the exact weight of a cow, they might be wildly incorrect. If you ask 500 students, and average their independent guesses perfectly, the aggregate answer will be incredibly close to reality. This is Ensemble Learning.

A Random Forest mathematically builds hundreds of separate Decision Trees. When generating a final prediction, every tree gets one "Vote". The majority class wins.

## Step 2: Bootstrap Aggregation (Bagging)

Why don't the 100 trees just mathematically yield the exact identical result? 

Because of **Bootstrap Aggregation (Bagging)** and **Feature Sampling**:
1. Every new tree is trained on a randomly selected subset of the rows.
2. Every mathematical node split inside the tree is only allowed to look at a randomly selected subset of the columns (features).

This forces geometric diversity natively.

## Step 3: Implementation

We will use Scikit-Learn to build a robust ensemble using the `iris` dataset.

```python
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Prepare standard data
df = sns.load_dataset('iris')
X = df.drop(columns='species')
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Instantiate powerful parameters natively
# n_estimators dictates exactly how many trees to explicitly build
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

# 3. Train all 100 trees explicitly
rf.fit(X_train, y_train)

# 4. Extract algorithmic voting predictions natively
preds = rf.predict(X_test)

print(f"Accuracy structurally: {accuracy_score(y_test, preds):.2f}")
```

??? example "Expected Output"
    ```text
    Accuracy structurally: 1.00
    ```

Random Forest natively handles complex multi-class data efficiently without requiring complex hyper-parameter tuning algebraically.

!!! tip "Workplace Tip"
    Random forests practically structurally do not overfit logically by directly adding more trees natively. Increasing `n_estimators` from 100 to 1,000 mathematically simply increases execution time, but explicitly mechanically stabilizes the voting variance cleanly. However, increasing `max_depth` will dynamically aggressively cause catastrophic overfitting.

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S13 | Apply ML algorithms | Operating Ensemble Random mathematical Forests |
| K5 | Machine Learning workflows | Managing hyperparameters avoiding Variance cleanly |
