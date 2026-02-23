# Creating Features from Raw Data

> "Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data." — Dr. Jason Brownlee

## What You Will Learn

- Engineer interaction terms to highlight latent relationships between columns
- Build polynomial features to map nonlinear dependencies
- Construct mathematical ratios
- Perform Continuous-to-Categorical binning

## Prerequisites

- [Pipelines in Data Prep](../../topic-1-data-preparation/tutorials/pipelines.md)

## Step 1: Mathematical Ratios & Interaction Terms

Algorithms are good at finding patterns *if* those patterns are structurally defined. If your dataset contains `Total_Debt` and `Total_Income`, the algorithm can learn from them independently, but explicitly defining the `Debt_to_Income_Ratio` provides immediate, non-linear context.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Sample banking data
data = pd.DataFrame({
    'Total_Debt': [50000, 10000, 200000, 5000],
    'Total_Income': [60000, 80000, 150000, 30000]
})

# Interaction terms (Ratios)
# Add a small epsilon (1e-6) to denominator to avoid division by zero
data['Debt_to_Income'] = data['Total_Debt'] / (data['Total_Income'] + 1e-6)

# Interaction terms (Multipliers)
# e.g., 'House_Width' * 'House_Length' = 'House_Square_Footage'

print(data)
```

## Step 2: Polynomial Features Workflow

For Linear Models (Linear Regression, Logistic Regression), relationships are assumed to be a straight line. If the actual pattern curves, a straight line will fail (Underfitting).

We can forcefully inject curved logic into basic line-fitting models using `PolynomialFeatures`.

\\[
y = \\beta_0 + \\beta_1(x_1) + \\beta_2(x_1^2) + \\epsilon
\\]

```python
from sklearn.preprocessing import PolynomialFeatures

# Creating an exponential feature visually
X = np.arange(1, 6).reshape(-1, 1)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Original X:\\n{X}")
print(f"\\nPolynomial X (x^1, x^2):\\n{X_poly}")
```

## Step 3: Binning (Discretization)

Sometimes algorithms perform better if you group continuous data into logical categories (e.g. converting `Age=34` into the category `30-40`). 

Trees implicitly do this, but Linear Models benefit greatly from explicit bins because it allows them to learn non-linear patterns over different intervals.

```python
from sklearn.preprocessing import KBinsDiscretizer

ages = np.array([22, 25, 34, 45, 55, 60, 21, 23, 80, 75]).reshape(-1, 1)

# Bin into 4 groups ('uniform' means equal width bins)
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
binned_ages = est.fit_transform(ages)

df_ages = pd.DataFrame({'Original_Age': ages.flatten(), 'Age_Group': binned_ages.flatten()})
print(df_ages.sort_values('Original_Age'))
```

!!! tip "Workplace Tip"
    In marketing or economics divisions, **Binning** is critical. You rarely market a product to "people aged exactly 34". You build campaigns for "Millennials (25-40)". Feature engineering directly translates real-world business segments into algorithm logic.

## Summary

Feature Engineering is the single most important mechanism for increasing the performance limit of your application. You are structurally manipulating the dimensionality of your model to expose facts that were previously buried. 

## Next Steps

→ [DateTime & Text Features](datetime-text-features.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S2 | Apply machine learning | Using Polynomial expansions |
| S4 | Transform data | Implements continuous-to-categorical grouping |
