# How-to: Handle Imbalanced Datasets

## The Problem
Your company wants to catch credit card fraudsters. 99% of transactions are legitimate. 1% are fraudulent. 

If you feed this directly into a Logistic Regression or Random Forest, the mathematical engine will simply realize that it can achieve 99% accuracy by guessing "Legitimate" 100% of the time. The algorithm is technically correct, but completely functionally useless. It has optimized out the 1% target.

## The Solution
You must artificially manipulate the dataset prior to model training so that both classes have equal footing mathematically.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Imblearn is a specialized library built explicitly for minority class manipulation
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Generate highly skewed data (95% Class 0 vs 5% Class 1)
X, y = make_classification(n_samples=5000, n_features=10, weights=[0.95, 0.05], random_state=42)

# MANDATORY: Split FIRST, Resample SECOND
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f"Original Training Distribution:\\n{pd.Series(y_train).value_counts()}")

# Approach 1: Random Under-Sampling
# Reduces the majority class down to randomly match the exact count of the minority class.
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)

# Approach 2: SMOTE (Synthetic Minority Over-sampling Technique)
# Synthesises NEW fake geometric data points near the existing minority class.
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
```

## Discussion

### The "Split First" Rule
> [!WARNING]
> If you apply SMOTE *before* you `train_test_split()`, you will synthetically generate fake data points across the entire feature space. When you split, some of these fake points will end up in your Testing Set. You will fundamentally evaluate an algorithm on fabricated targets, ensuring a dramatic real-world failure.

### When to Undersample vs Oversample?
- **Undersampling (RUS)** destroys data. If you have 10,000 majority rows and 50 minority rows, doing RUS deletes 9,950 rows of data. It is only viable on absolutely colossal datasets where massive computation is a bottleneck.
- **Oversampling (SMOTE)** is structurally superior for small datasets, though it massively inflates training geometry, requiring substantially higher memory allocations.
