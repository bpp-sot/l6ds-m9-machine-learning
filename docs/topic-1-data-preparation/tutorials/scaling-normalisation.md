# Scaling & Normalisation

> "Distance-based algorithms behave like real estate agents—they calculate distance based on whatever scale you give them. A difference of 100,000 in salary shouldn't outweigh a difference of 30 in age."

## What You Will Learn

- Understand why feature scaling is critical for certain algorithms
- Implement `StandardScaler` (z-score normalization)
- Implement `MinMaxScaler`
- Use `RobustScaler` to handle datasets with significant outliers

## Prerequisites

- [Data Types & Encoding](data-types-encoding.md)

## Step 1: Why Scale?

Algorithms that compute distance (KNN, K-Means, SVM) or use Gradient Descent (Neural Networks, Linear Regression) require features to be on the same scale. The mathematical derivation relies on updating weights symmetrically.

\\[
z = \\frac{x - \\mu}{\\sigma}
\\]
*(The formula for Standardization)*

## Step 2: Standardization (StandardScaler)

Standardization centers your data around 0 with a standard deviation of 1.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Generate dummy data: Age (small range) and Salary (large range)
np.random.seed(42)
ages = np.random.normal(35, 10, 1000)
salaries = np.random.normal(60000, 15000, 1000)
df = pd.DataFrame({'Age': ages, 'Salary': salaries})

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x='Age', y='Salary', data=df, ax=ax1)
ax1.set_title('Original Scale')

sns.scatterplot(x='Age', y='Salary', data=df_scaled, ax=ax2)
ax2.set_title('StandardScaler Applied')
plt.show()
```

!!! tip "Workplace Tip"
    `StandardScaler` is generally the default choice. If you don't know which scaler to use, start here.

## Step 3: Normalization (MinMaxScaler)

Min-Max Scaling squashes values into a fixed range, usually `[0, 1]`.

```python
from sklearn.preprocessing import MinMaxScaler

min_max = MinMaxScaler()
df_minmax = pd.DataFrame(min_max.fit_transform(df), columns=df.columns)

print("MinMax Summary:")
print(df_minmax.describe().round(2))
```

This is heavily utilized in deep learning (image pixel data) and when you must strictly bound your variables.

## Step 4: RobustScaler (Outlier Resistance)

If your dataset is prone to extreme outliers, `StandardScaler` will severely deform your distribution because outliers pull the mean and standard deviation. The `RobustScaler` uses the **median** and the **Interquartile Range (IQR)** instead.

```python
from sklearn.preprocessing import RobustScaler

# Inject severe outliers
df.loc[0:10, 'Salary'] = df.loc[0:10, 'Salary'] * 10 

robust = RobustScaler()
df_robust = pd.DataFrame(robust.fit_transform(df), columns=df.columns)
```

## Summary

Scaling techniques protect your model. Use `MinMaxScaler` for fixed bounds, `StandardScaler` for normally distributed features, and `RobustScaler` when anomalies corrupt your parameters.

## Next Steps

→ [Detecting & Treating Outliers](outliers.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S2 | Apply machine learning techniques | Prepares gradients for convergence |
| K1 | Statistical Concepts | Implements standard deviation and IQR transformations |
