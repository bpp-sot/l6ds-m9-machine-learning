# Regularisation Explained

> In predictive modelling, features often "shout over" each other. Regularisation forces the algorithm to turn down the volume on overly dominant coefficients, preventing overfitting.

## Why Regularise?

Without constraints, a linear model will assign whatever coefficient values minimise training error — even if that means inflating weights to extreme values that capture noise rather than signal. Regularisation adds a **penalty term** to the loss function that punishes large coefficients.

## The L1 Penalty (Lasso)

L1 regularisation adds the **sum of absolute coefficient values** to the loss function.

$$\text{Loss} = \text{MSE} + \alpha \sum |w_i|$$

**Key behaviour:** L1 drives weak or irrelevant coefficients all the way to exactly `0.0`. This means Lasso performs **automatic feature selection** — it physically eliminates useless features from the model.

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)  # alpha controls penalty strength
lasso.fit(X_train, y_train)

# Inspect which features were eliminated
import pandas as pd
pd.Series(lasso.coef_, index=feature_names).sort_values()
```

**Use when:** You suspect many features are irrelevant and want the model to select the important ones automatically.

## The L2 Penalty (Ridge)

L2 regularisation adds the **sum of squared coefficient values** to the loss function.

$$\text{Loss} = \text{MSE} + \alpha \sum w_i^2$$

**Key behaviour:** L2 shrinks all coefficients towards zero but never sets any to exactly `0.0`. It keeps all features in the model but dampens the influence of correlated or noisy ones.

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

**Use when:** You believe most features carry some signal and want to distribute weight evenly rather than eliminate features.

## Elastic Net (L1 + L2)

Elastic Net combines both penalties, giving you a tuneable ratio between feature selection (L1) and weight dampening (L2).

```python
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50/50 mix of L1 and L2
enet.fit(X_train, y_train)
```

## Choosing Alpha

The `alpha` hyperparameter controls the strength of regularisation. Use cross-validated variants to find the optimal value automatically:

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5).fit(X_train, y_train)
print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
```

!!! tip "Workplace Tip"
    Always standardise your features before applying regularisation. If features are on different scales, the penalty will disproportionately affect those with smaller magnitudes.

## KSB Mapping

| KSB | Description | How This Explanation Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Understanding how penalty terms control model complexity |
