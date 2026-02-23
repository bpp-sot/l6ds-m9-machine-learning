# How-to: Build a Baseline Model

## The Problem
You spent 3 weeks building a highly complex, 200-layer Neural Network that predicts customer churn with 85% accuracy. Is that "good"? 

If 85% of your customers rarely churn anyway, a model that simply predicts "They will not churn" without doing any math will also be 85% accurate. Your Neural Network is mathematically worthless.

## The Solution
Always build a "Dummy" baseline model *before* building any machine learning algorithms. The baseline provides the "Zero Value" floor. If your XGBoost model cannot beat the Dummy model, you do not have a Machine Learning solution.

### 1. Classification Baselines

`DummyClassifier` simply guesses the most frequent class, or guesses randomly based on class distribution.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 90% of data is Class 0 (Massively Imbalanced)
y_train = np.array([0]*900 + [1]*100)
X_train = np.random.rand(1000, 5) # Features don't matter, Dummy ignores them

y_test = np.array([0]*90 + [1]*10)
X_test = np.random.rand(100, 5)

# Strategy: Always guess the most frequent class (Class 0)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

baseline_preds = dummy_clf.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)

print(f"Baseline (Zero-Math) Accuracy: {baseline_acc * 100:.1f}%")
print("If your Random Forest does not exceed 90.0%, it is utterly useless.")
```

### 2. Regression Baselines

`DummyRegressor` simply guesses the average (mean or median) historical value for every single prediction.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

# Historical house prices
y_train_prices = np.array([100k, 150k, 200k, 250k, 300k])
y_test_prices = np.array([180k, 220k])

# Strategy: Always guess the Mean Training Price (200k)
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(None, y_train_prices)

baseline_price_preds = dummy_reg.predict(None)

print(f"Test Actuals: {y_test_prices}")
print(f"Baseline Guesses: {baseline_price_preds}")
```

## Discussion

In the L6 Assessment presentation, establishing a "Baseline" is specifically enumerated in the rubric markers. Demonstrating that your engineered pipeline generates exactly X% improvement over the mathematical floor is how you convert academic accuracy into workplace ROI.
