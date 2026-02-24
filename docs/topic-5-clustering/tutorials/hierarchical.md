# Hierarchical Clustering

> Bottom-up agglomerative clustering builds a tree (dendrogram) by iteratively merging the two closest clusters until only one remains.

## How It Works

1. Start with each data point as its own cluster.
2. Find the two closest clusters and merge them.
3. Repeat until all points belong to a single cluster.
4. Cut the dendrogram at the desired height to obtain your clusters.

The **linkage criterion** determines how "closeness" between clusters is measured:

| Linkage | Measures Distance Between |
|---------|--------------------------|
| `ward` | Minimises variance increase when merging (default, produces compact clusters) |
| `complete` | Farthest points in each cluster |
| `average` | Mean distance between all point pairs |
| `single` | Nearest points in each cluster (prone to chaining) |

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.8, random_state=42)

# 1. Plot the Dendrogram
Z = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# 2. Fit Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2", edgecolor="k", s=40)
plt.title("Agglomerative Clustering (3 Clusters)")
plt.tight_layout()
plt.show()
```

## When to Use Hierarchical Clustering

- When you want a **visual overview** of how clusters relate (dendrogram).
- When you do not know the optimal number of clusters in advance — the dendrogram helps you decide.
- When cluster sizes are small to medium (hierarchical clustering scales as \(O(n^2)\) in memory).

!!! warning "Common Pitfall"
    Hierarchical clustering does **not** scale well. For datasets larger than ~10,000 points, prefer k-Means or DBSCAN.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Building and interpreting hierarchical cluster structures |
