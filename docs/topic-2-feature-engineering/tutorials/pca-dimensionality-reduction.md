# PCA & Dimensionality Reduction

> "When data lives in a hundred dimensions, human intuition completely breaks down. PCA is our mathematical flashlight into the dark."

## What You Will Learn

- Understand Principal Component Analysis (PCA) conceptually
- Transform highly correlated features into independent Principal Components
- Select `n_components` empirically using Scree Plots and Explained Variance Ratio

## Prerequisites

- [Scaling & Normalisation](../../topic-1-data-preparation/tutorials/scaling-normalisation.md)

## Step 1: Concept Overview

When you have hundreds of features (e.g., measuring house prices using `square_footage`, `number_of_bedrooms`, `number_of_bathrooms`, `lot_size`), many of those features say the *exact same thing*. 

If you know a house is 5,000 square feet, you already know it probably has more than 1 bathroom. This redundancy is mathematically inefficient.

**PCA** identifies the axes (directions) in your data that contain the most variance (information) and physically rotates the dataset onto those new axes, dropping the axes that contain nothing but noise.

```mermaid
graph LR
    A[Original Features] -->|Scaling| B[Centered Data]
    B -->|Calculate Eigenvectors| C[Principal Components]
    C -->|Project Data| D[Reduced Feature Space]
```

## Step 2: Implementation

> [!CAUTION]
> PCA is exceptionally sensitive to scale! If you apply PCA before `StandardScaler`, PCA will simply identify your largest-scaled feature as the "First Principal Component" and fail catastrophically.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style('whitegrid')

# Load dataset (30 features!)
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# 1. MUST SCALE FIRST
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit PCA (Let's see all components first)
pca = PCA()
pca.fit(X_scaled)
```

## Step 3: Determining `n_components`

How many components should you keep? We look at the **Explained Variance Ratio**.

```python
# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(15, 0.85, '90% Variance Threshold', color='red', fontsize=12)
plt.title('PCA Scree Plot (Cumulative Explained Variance)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Find exactly how many components we need to reach 90%
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"We can reduce the dataset from {X.shape[1]} features down to {n_components_90} features while keeping 90% of the information!")
```

## Step 4: The Final Transformation

Once you determine the threshold, apply the transformation.

```python
# Re-instantiate with target components
final_pca = PCA(n_components=n_components_90)
X_pca = final_pca.fit_transform(X_scaled)

# Creating a 2D scatter plot using the first two principal components
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('2D PCA Projection of Breast Cancer Data')
plt.legend(handles=scatter.legend_elements()[0], labels=['Malignant', 'Benign'])
plt.show()
```

## Summary

PCA compresses information. It solves the Curse of Dimensionality by dropping noise and highly collinear artifacts, resulting in faster and more stable machine learning executions.

## Next Steps

Explore the How-To guides in this section to apply Feature Engineering logic contextually.

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| K1 | Statistical Concepts | Implements eigenvectors mathematically |
| S7 | Generate Insights | Visually flattens 30 dimensions down to a 2D plot for stakeholder analysis |
