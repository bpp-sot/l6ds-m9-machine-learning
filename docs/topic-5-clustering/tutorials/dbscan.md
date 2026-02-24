# DBSCAN & Density-Based Methods

> Density-Based Spatial Clustering of Applications with Noise (DBSCAN) finds clusters of arbitrary shape by grouping points in dense regions and labelling sparse points as noise.

## How It Works

DBSCAN requires two parameters:

| Parameter | Meaning |
|-----------|---------|
| `eps` | Maximum distance between two points to be considered neighbours |
| `min_samples` | Minimum number of points required to form a dense region (core point) |

The algorithm classifies each point as:

- **Core point:** Has at least `min_samples` neighbours within `eps`.
- **Border point:** Within `eps` of a core point but has fewer than `min_samples` neighbours.
- **Noise point:** Neither core nor border — labelled as `-1`.

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2", edgecolor="k", s=30)
plt.title(f"DBSCAN — {len(set(labels) - {-1})} clusters, {(labels == -1).sum()} noise points")
plt.tight_layout()
plt.show()
```

## Advantages Over k-Means

| Feature | k-Means | DBSCAN |
|---------|---------|--------|
| Cluster shape | Spherical only | Arbitrary shapes |
| Requires \(k\) | Yes | No — discovers clusters automatically |
| Noise handling | None — assigns every point | Labels outliers as `-1` |
| Density variation | Cannot handle | Struggles with varying densities |

## Tuning `eps`

Use a **k-distance plot** to estimate `eps`:

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# Sort the distance to the 5th nearest neighbour
sorted_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(sorted_distances)
plt.xlabel("Points (sorted)")
plt.ylabel("5th Nearest Neighbour Distance")
plt.title("k-Distance Plot — Look for the Elbow")
plt.tight_layout()
plt.show()
```

!!! tip "Workplace Tip"
    DBSCAN excels when your data has natural clusters of irregular shape (e.g., geographic regions, network communities). If clusters have very different densities, consider `HDBSCAN` instead.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Applying density-based clustering for arbitrary-shaped groups |
