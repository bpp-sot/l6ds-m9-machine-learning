# Domain Knowledge vs. Automated Engineering

> Is it better to manually craft features using business expertise, or automatically generate thousands of candidate features using brute force?

## The Case for Domain Knowledge

If you work in healthcare, a clinician knows that `BMI` biologically correlates with diabetes risk.

You extract it directly:

```python
df["bmi"] = df["weight"] / df["height"] ** 2
```

**Pros:**

- The resulting feature is transparent and interpretable to stakeholders.
- Domain-driven features are highly resistant to overfitting because they encode a genuine causal or correlational relationship.

**Cons:**

- Requires access to a subject-matter expert.
- You can only create features you already know about — you will miss unexpected interactions.

## The Case for Automation

Libraries like `FeatureTools` programmatically generate every possible arithmetic combination of your columns (sums, products, ratios, squares).

```python
engineered = ["age * income", "age / distance", "distance ** 2"]
```

**Pros:**

- Discovers non-linear relationships that no human would think to test.
- Scales effortlessly to wide datasets with dozens of raw columns.

**Cons:**

- Produces a vast number of features, most of which are noise — increasing overfitting risk.
- Generated features are opaque and harder to justify in a business context.

## The Hybrid Approach

In practice, you combine both strategies:

1. **Start with domain knowledge.** Engineer the features your business logic demands (e.g., `tenure_months`, `spend_per_visit`).
2. **Augment with automation.** Use `PolynomialFeatures` or `FeatureTools` to generate interaction terms, then apply feature selection (e.g., mutual information, recursive feature elimination) to discard the noise.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)
```

This gives you interpretable core features enriched by algorithmically discovered interactions — the best of both worlds.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Feature selection algorithms and dimensionality reduction |
| K5.2 | Data formats and structures | Encoding categorical variables, handling mixed feature types |
| S2 | Data engineering | Creating and transforming features from raw data |
| S4 | Feature selection and ML | Applying feature selection methods and PCA |
| B1 | Inquisitive approach | Exploring creative feature engineering strategies |
