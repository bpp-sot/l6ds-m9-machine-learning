# How to Build a Baseline Model

> Before building a mathematically complex Neural Network, you must always establish a "Baseline" score. A Baseline is the absolutely simplest possible prediction you could logically make.

## Why Baselines Matter

If your Random Forest scores `85%` Accuracy, is that good? 

* If you are predicting a coin flip, `85%` is incredible.
* If you are predicting whether an asteroid will hit Earth today (where 99.9999% of the time it doesn't), `85%` accuracy is catastrophic.

A baseline proves if your Machine Learning algorithm is actually *learning* anything at all.

## Step 1: The Dummy Classifier (Classification)

Scikit-Learn provides `DummyClassifier`. It completely ignores your $X$ variables natively and blindly guesses the mathematical mode (the most frequent class) inherently.

```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# 1. Load an imbalanced dataset conceptually
df = sns.load_dataset('diamonds')

# Predict if cut is explicitly "Fair" (only ~3% of dataset)
df['is_fair'] = (df['cut'] == 'Fair').astype(int)

X = df[['carat', 'depth', 'price']]
y = df['is_fair']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Instantiate DummyClassifier cleanly
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

# 3. Generate baseline accuracy natively
baseline_preds = dummy.predict(X_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, baseline_preds):.3f}")
```

??? example "Expected Output"
    ```text
    Baseline Accuracy: 0.970
    ```

By literally ignoring all data and just globally guessing "Not Fair", you mathematically achieve `97.0%` accuracy explicitly! If your subsequent neural network achieves `96%`, your complex model is quite literally logically worse than a blind guess natively.

## Step 2: The Dummy Regressor (Regression)

For continuous numbers dynamically, we utilize the `DummyRegressor` to map baseline scores.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error

# Predict price logically
X_reg = df[['carat', 'depth']]
y_reg = df['price']

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, random_state=42)

# It ignores X, and simply mathematically averages y_train completely.
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_tr, y_tr)

reg_preds = dummy_reg.predict(X_te)
print(f"Baseline RMSE score: ${root_mean_squared_error(y_te, reg_preds):.2f}")
```

??? example "Expected Output"
    ```text
    Baseline RMSE score: $3984.34
    ```

If your subsequent Linear Regression doesn't achieve an RMSE lower than `$3,984`, your mathematical features are effectively useless dynamically.

!!! tip "Workplace Tip"
    Always report your final algorithm metrics directly beside the baseline. "Our Random Forest achieved 92% Accuracy, which represents a strictly +14% absolute mathematical uplift analytically over the Dummy Baseline explicitly."

## KSB Mapping

| KSB | Description | How This Guide Addresses It |
|-----|-------------|----------------------------|
| B2 | Analytical | Proving algorithmic legitimacy systematically |
| K5 | Machine Learning workflows | Establishing control metrics defensively |
