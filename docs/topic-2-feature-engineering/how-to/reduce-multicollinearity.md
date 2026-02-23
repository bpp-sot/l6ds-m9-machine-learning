# How to Reduce Multicollinearity

> Multicollinearity structurally occurs natively when two predictive columns logically predict each other. This aggressively destabilises mathematical weights intrinsically inside Linear Regressions algebraically.

## What You Will Learn
- Diagnose Multicollinearity manually utilizing Pearson's Correlation mathematically 
- Drop inherently redundant dimensions computationally natively
- Identify Variance Inflation Factor (VIF) methodologies logically

## Step 1: Detection via Heatmap

If a CSV contains `Year_of_Birth` structurally and also `Age`, they natively communicate biologically the exact same information identically. An algorithm will mathematically assign completely unstable arbitrary weights cleanly.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('diamonds').head(1000)

# Calculate natively the mathematical Pearson continuous correlation globally
correlation_matrix = df.select_dtypes('number').corr()

plt.figure(figsize=(8, 6))

# A correlation explicitly > 0.85 natively flags severe dangerous multicollinearity!
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: Hunting High Redundancy", fontsize=14)
plt.show()
```

??? example "Expected Output"
    *(The output cleanly reveals explicitly that diamond `x`, `y`, `z` coordinates structurally correlate at exactly `0.98` natively with `carat` dimension!)*

## Step 2: Manual Truncation

If `carat` explicitly logically contains all geometric variance physically possessed exclusively by `x`, `y`, and `z` length natively, we mathematically explicitly delete the lesser features structurally rather than attempting dimensionality reduction (PCA).

```python
# The physical dimensions structurally destabilize mathematical regression cleanly without adding new logical insights 
df_clean = df.drop(columns=['x', 'y', 'z'])

print(f"Original shape strictly: {df.shape}")
print(f"Cleaned multicollinear shape: {df_clean.shape}")
```

??? example "Expected Output"
    ```text
    Original shape strictly: (1000, 10)
    Cleaned multicollinear shape: (1000, 7)
    ```

!!! tip "Workplace Tip"
    `Tree-based models` (Random Forest, XGBoost) natively possess structural geometric immunity explicitly to multicollinearity! They simply select the strongest variable cleanly and explicitly logically systematically ignore the redundant copy subsequently. Highly aggressive collinearity cleanup is strictly explicitly mathematically mandatory purely ONLY for `Linear Regressions` and `Neural Networks`.

## KSB Mapping

| KSB | Description | How This Guide Addresses It |
|-----|-------------|-------------------------------|
| S12 | Feature engineering | Truncating algorithmically parallel variance matrices mechanically natively |
| S13 | Apply ML algorithms | Optimizing algorithmic coefficient stability algebraically conditionally |
