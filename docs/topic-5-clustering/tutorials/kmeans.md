# k-Means Clustering

> The foundational partitioning algorithm that assigns each data point to the nearest of \(k\) centroids, then iteratively refines centroid positions until convergence.

## How It Works

1. Randomly initialise \(k\) centroids.
2. **Assign** each point to its nearest centroid (Euclidean distance).
3. **Update** each centroid to the mean of its assigned points.
4. Repeat steps 2–3 until centroids stop moving (convergence).

## Implementation

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic clustered data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Fit k-Means
km = KMeans(n_clusters=4, random_state=42, n_init="auto")
labels = km.fit_predict(X)

# Visualise
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2", edgecolor="k", s=30)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("k-Means Clustering (k=4)")
plt.legend()
plt.tight_layout()
plt.show()
```

## Key Properties

| Property | Detail |
|----------|--------|
| **Cluster shape** | Assumes spherical, equally-sized clusters |
| **Requires \(k\)** | You must specify the number of clusters in advance |
| **Sensitivity** | Sensitive to initialisation — use `n_init="auto"` for multiple runs |
| **Feature scaling** | Mandatory — Euclidean distance is scale-dependent |
| **Convergence** | Always converges, but may find a local minimum |

## Limitations

- Cannot detect clusters of arbitrary shape (use DBSCAN for that).
- Outliers pull centroids away from true cluster centres.
- All clusters tend towards equal size due to Voronoi partitioning.

!!! tip "Workplace Tip"
    k-Means is your first port of call for clustering. It is fast, well-understood, and works well when clusters are roughly spherical. Always standardise features first with `StandardScaler`.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Implementing the foundational unsupervised clustering algorithm |
