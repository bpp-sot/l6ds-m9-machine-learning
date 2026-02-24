# The Curse of Dimensionality

> As you add more columns to your dataset, the volume of the feature space explodes exponentially.

## The Intuition

Imagine dropping a single coin onto a straight 10-metre line. You locate it almost immediately.

Now, imagine dropping that coin randomly into a 10m × 10m × 10m warehouse. Locating it requires thousands of independent searches.

This is exactly what happens to your ML algorithm as you add dimensions.

## The Mathematical Breakdown

Distance-based algorithms such as `K-Means` and `KNN` rely on Euclidean Distance to measure similarity between observations.

If you One-Hot Encode a categorical feature containing 500 distinct cities, you add exactly 500 new axes to your feature space.

With 500 dimensions, the distance between *every* pair of observations converges towards the same large value. When every row is equidistant from every other row, distance-based algorithms lose all discriminating power and fail to cluster or classify. This is the **Curse of Dimensionality**.

## Feature Engineering Impact

**Do not One-Hot Encode high-cardinality features.**

If `Postcode` contains 1,500 distinct values, One-Hot Encoding will produce a 1,500-column sparse matrix that cripples your model.

Instead, use **Target Encoding** to compress a high-cardinality string column into a single numeric column representing the mean of the target variable per category.

```python
from sklearn.preprocessing import TargetEncoder

enc = TargetEncoder()
df["postcode_encoded"] = enc.fit_transform(df[["postcode"]], df["target"])
```

!!! tip "Workplace Tip"
    PCA (Principal Component Analysis) exists specifically to combat the Curse of Dimensionality. It projects your high-dimensional data onto a smaller set of orthogonal components that capture the maximum variance, reducing column count while preserving signal.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Feature selection algorithms and dimensionality reduction |
| K5.2 | Data formats and structures | Encoding categorical variables, handling mixed feature types |
| S2 | Data engineering | Creating and transforming features from raw data |
| S4 | Feature selection and ML | Applying feature selection methods and PCA |
| B1 | Inquisitive approach | Exploring creative feature engineering strategies |
