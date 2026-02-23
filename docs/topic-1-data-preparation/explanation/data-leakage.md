# Data Leakage

> Data leakage is the most lethal mistake in Machine Learning. It occurs when information from outside the training dataset infects the model during training, generating wildly optimistic but physically impossible deployment results.

## The Concept

Imagine a teacher giving students a maths exam. If the teacher accidentally leaves the answer key on the projector while the students are studying, every student will score 100%. But when those students are sent out into the real world to build a bridge, the bridge collapses because they never naturally learned the mathematics; they just memorised the leaked answers.

In Machine Learning, if your algorithm "sees" any structural information from the `Test Data` during the `Training Phase`, it has cheated.

## The Most Common Cause: Scaling Before Splitting

The most frequent way Juniors introduce leakage is by calling `.fit_transform()` on the entire dataset *before* utilizing `train_test_split()`.

```python
# 🔴 DANGEROUS LEAKAGE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
# The scaler mathematically computes the exact 'mean' of the ENTIRE dataset!
X_scaled = scaler.fit_transform(X) 

# Later, you split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

Why is this broken? The `StandardScaler` calculated the average of the *Test Set* and used it to scale the *Training Set*. Your algorithm now secretly possesses the global mathematical properties of the future invisible testing data!

## The Solution: Split, Fit, Transform

You must securely quarantine your `Test Set` dynamically into a "vault" before executing any preprocessing logic. 

```python
# 🟢 SECURE PREPROCESSING
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Quarantine the data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

# 2. The scaler learns the mathematical mean ONLY from X_train
X_train_scaled = scaler.fit_transform(X_train) 

# 3. We blindly apply that specific mean scalar onto the unseen Test Set
X_test_scaled = scaler.transform(X_test) 
```

Notice we explicitly execute `.transform()` on `X_test`, not `.fit_transform()`. We absolutely do not want the scaler "learning" the arithmetic mean of the testing array!

!!! info "Assessment Connection"
    If your EPA predictive model generates an impossibly flawless 99.9% accuracy score during local Validation, your examiner will aggressively hunt your code for Data Leakage. Explicitly stating that you utilized isolated `sklearn.pipeline.Pipeline` objects to systematically eliminate Test Set leakage proves senior-level engineering competence.

## KSB Mapping

| KSB | Description | How This Explanation Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Understanding and eliminating cross-validation contamination structures |
| B2 | Logical and analytical approach | Architecting rigorous uncoupled execution boundaries defensively |
