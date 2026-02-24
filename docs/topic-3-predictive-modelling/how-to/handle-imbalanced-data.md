# How to Handle Imbalanced Data

> When one class dominates your dataset (e.g., 99% Non-Fraud, 1% Fraud), standard ML algorithms bias entirely toward the majority class.

## Why Imbalance is Dangerous

If a model blindly predicts "Non-Fraud" for every single transaction, it achieves 99% Accuracy. However, detecting fraud is the entire business objective. Standard algorithms minimise overall error, sacrificing the minority class entirely to achieve high global accuracy.

## Method 1: Algorithmic Class Weights

The simplest solution is adjusting `class_weight`. Most scikit-learn classifiers accept `"balanced"`, which automatically increases the penalty for misclassifying minority observations.

```python
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = sns.load_dataset("diamonds").sample(1000, random_state=42)
X = df[["carat", "depth", "table"]]
y = (df["cut"] == "Premium").astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))
```

## Method 2: Synthetic Minority Oversampling (SMOTE)

Instead of reweighting the algorithm, you synthetically generate new minority-class data points by interpolating between existing minority observations.

You need to install the `imbalanced-learn` library: `pip install imbalanced-learn`.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Original training shape: {X_train.shape}")
print(f"Resampled training shape: {X_train_res.shape}")
```

!!! warning "Critical Rule"
    **Never apply SMOTE before `train_test_split`.** If you synthesise data before splitting, synthetic copies of minority observations will leak into the test set, giving you artificially inflated scores that do not reflect real-world performance.

## Method 3: Threshold Tuning

By default, classifiers use a 0.5 probability threshold. For imbalanced problems, lowering the threshold (e.g., to 0.3) increases Recall at the cost of Precision.

```python
import numpy as np

probs = model.predict_proba(X_te)[:, 1]
custom_preds = (probs >= 0.3).astype(int)
print(classification_report(y_te, custom_preds))
```

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-----------------------|
| S15 | Evaluate model performance | Implementing strategies to handle class imbalance |
