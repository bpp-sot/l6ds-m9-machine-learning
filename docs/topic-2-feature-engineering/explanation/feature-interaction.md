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

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Feature selection algorithms and dimensionality reduction |
| K5.2 | Data formats and structures | Encoding categorical variables, handling mixed feature types |
| S2 | Data engineering | Creating and transforming features from raw data |
| S4 | Feature selection and ML | Applying feature selection methods and PCA |
| B1 | Inquisitive approach | Exploring creative feature engineering strategies |
