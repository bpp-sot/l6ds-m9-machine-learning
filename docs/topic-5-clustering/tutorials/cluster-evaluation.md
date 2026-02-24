# Evaluating Cluster Quality

> Unlike classification, clustering has no ground truth. You must evaluate using internal validity metrics like the Silhouette Score.

## Internal Metrics

When true labels are unknown, we rely on intrinsic measures that assess how well-separated and compact the clusters are.

### Silhouette Score

For each point, the Silhouette Score compares its average distance to points in its own cluster (\(a\)) versus the nearest neighbouring cluster (\(b\)):

$$s = \frac{b - a}{\max(a, b)}$$

- **+1:** Point is far from neighbouring clusters (ideal).
- **0:** Point is on the boundary between clusters.
- **-1:** Point is likely in the wrong cluster.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

km = KMeans(n_clusters=4, random_state=42, n_init="auto")
labels = km.fit_predict(X)

score = silhouette_score(X, labels)
print(f"Mean Silhouette Score: {score:.3f}")

# Per-sample silhouette values
sample_scores = silhouette_samples(X, labels)
print(f"Min: {sample_scores.min():.3f}, Max: {sample_scores.max():.3f}")
```

### Calinski-Harabasz Index

Measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.

```python
from sklearn.metrics import calinski_harabasz_score

ch = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz: {ch:.1f}")
```

### Davies-Bouldin Index

Measures the average similarity between each cluster and its most similar cluster. Lower is better (0 = perfect separation).

```python
from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin: {db:.3f}")
```

## Summary Table

| Metric | Range | Goal | Needs True Labels? |
|--------|-------|------|-------------------|
| Silhouette Score | [-1, 1] | Maximise | No |
| Calinski-Harabasz | [0, ∞) | Maximise | No |
| Davies-Bouldin | [0, ∞) | Minimise | No |

!!! warning "Common Pitfall"
    These metrics favour convex, globular clusters. They may give misleading results for non-convex shapes (e.g., crescent-shaped clusters found by DBSCAN).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
