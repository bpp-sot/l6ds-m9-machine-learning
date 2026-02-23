# Handling Missing Values

> "A data scientist's most common phrase is 'it depends'. Their second is 'where's the missing data?'"

## What You Will Learn

- Identify different types of missing data mechanisms (MCAR, MAR, MNAR)
- Use basic imputation strategies (`SimpleImputer`)
- Implement advanced imputation methods (`KNNImputer`, `IterativeImputer`)
- Evaluate the impact of imputation choices on data distribution

## Prerequisites

- [Loading & Exploring Data](loading-exploring.md)
- Familiarity with the `scikit-learn` API structure

## Step 1: Diagnosing Missingness

Before choosing an imputation strategy, you must understand *why* the data is missing. Here we will synthesize a dataset with missing values to demonstrate.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set consistent styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load sample data and inject missing values
data = fetch_california_housing(as_frame=True).frame
# Simulate Missing Completely At Random (MCAR)
mask = np.random.rand(*data.shape) < 0.1
missing_data = data.mask(mask)

print(f"Missing Values Check:\\n{missing_data.isnull().sum()}")
```

## Step 2: Simple Imputation 

The `SimpleImputer` replaces missing values with univariate statistics like mean, median, or constant values.

```python
from sklearn.impute import SimpleImputer

# Separate features and target
X = missing_data.drop('MedHouseVal', axis=1)
y = missing_data['MedHouseVal']

# Impute with median
imputer = SimpleImputer(strategy='median')
X_median_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Visualizing the shift in distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(X['AveRooms'], ax=ax[0], label='Original with Missing', color='blue')
sns.kdeplot(X_median_imputed['AveRooms'], ax=ax[0], label='Median Imputed', color='red')
ax[0].set_title('Kernel Density - AveRooms')
ax[0].legend()

sns.boxplot(data=[X['AveRooms'].dropna(), X_median_imputed['AveRooms']], ax=ax[1])
ax[1].set_xticklabels(['Original (Drop NA)', 'Median Imputed'])
ax[1].set_title('Boxplot Comparison')
plt.tight_layout()
plt.show()
```

!!! warning "Artifacts from Simple Imputation"
    Notice that median imputation can cause sharp spikes in the distribution exactly at the median value, artificially reducing the variance of your dataset.

## Step 3: Multivariate Imputation

If your features are correlated, you can use other features to predict the missing values. 

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# 1. K-Nearest Neighbors Imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)

# 2. Iterative Imputation (MICE)
iter_imputer = IterativeImputer(random_state=42, max_iter=10)
X_iter = pd.DataFrame(iter_imputer.fit_transform(X), columns=X.columns)
```

!!! tip "Workplace Tip"
    In industry, `IterativeImputer` often provides the most robust results for tabular data, but it is computationally expensive. If you need real-time inference in production, `SimpleImputer` is usually preferred for its speed.

## Summary

In this tutorial, you learned:
- How to diagnose missing data patterns
- Implementation of baseline univariate imputation
- Implementation of advanced multivariate imputation combining `KNNImputer` and `IterativeImputer`.

## Next Steps

→ [Data Types & Encoding](data-types-encoding.md) — now that missing values are handled, we must convert all text features to numeric representations.

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S4 | Import, cleanse, transform data | Demonstrates Scikit-Learn missing value handlers |
| K2 | Machine learning algorithms | Applies KNN conceptually to dataset fixing |
| B2 | Logical approach | Following a sequential pattern for data cleaning |
