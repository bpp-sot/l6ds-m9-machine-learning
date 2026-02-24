# Feature Interaction

> Feature interaction occurs when the predictive value of one feature depends entirely on the value of another.

## The Concept

Imagine you are predicting `House Prices`.

You have two independent features:

1. `Contains_Swimming_Pool` (Binary: Yes / No)
2. `Geographic_Location` (String: "Alaska" vs "Florida")

A model might learn that a swimming pool adds £10,000 to a property globally. However, a pool in *Florida* adds £15,000, while a pool in *Alaska* actively *reduces* the property value because of maintenance costs in a freezing climate.

The predictive effect of `Swimming_Pool` is conditional on `Location`. That conditional relationship is a **feature interaction**.

## The Algorithmic Failure

Linear algorithms (`LogisticRegression`, `LinearRegression`, `SVM`) assign a single flat coefficient to each feature. They cannot detect interaction effects natively — they will simply learn one global weight for `Swimming_Pool`, missing the geographic context entirely.

Tree-based algorithms (`DecisionTree`, `RandomForest`, `XGBoost`) handle interactions naturally by splitting on one feature and then splitting on another within the same branch.

## The Engineering Solution

To allow linear models to capture interactions, you must explicitly engineer the cross-product as a new column:

```python
df["pool_in_florida"] = df["contains_swimming_pool"] * df["location_florida"]
```

Alternatively, use scikit-learn's `PolynomialFeatures` to generate all pairwise interactions automatically:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interact = poly.fit_transform(X)
```

Now the linear model has a dedicated coefficient for the pool-in-Florida interaction, enabling it to learn the conditional effect.

!!! info "Assessment Connection"
    In your EPA, document *why* you multiplied features together. Stating that you engineered interaction terms to capture conditional relationships directly demonstrates the analytical maturity required by **B2 — Logical and Analytical Approach**.

## KSB Mapping

| KSB | Description | How This Explanation Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Understanding when and why interaction terms are necessary |
| B2 | Logical and analytical approach | Explicitly engineering features based on domain reasoning |
