# Find the Optimal Number of Clusters

> Use the Elbow Method and Silhouette Score together to select the right number of clusters for k-Means.

## The Elbow Method

Plot the **inertia** (within-cluster sum of squares) for increasing values of \(k\). The "elbow" — where the rate of decrease sharply levels off — suggests the optimal cluster count.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.show()
```

## The Silhouette Score

The Silhouette Score measures how similar each point is to its own cluster compared to neighbouring clusters. Values range from -1 (wrong cluster) to +1 (well-clustered).

```python
from sklearn.metrics import silhouette_score

scores = []
K_range = range(2, 11)  # Silhouette needs at least 2 clusters

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))

plt.figure(figsize=(8, 4))
plt.plot(K_range, scores, "ro-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.tight_layout()
plt.show()
```

!!! warning "Common Pitfall"
    The Elbow Method is subjective — the "elbow" is not always obvious. Always validate with the Silhouette Score and domain knowledge.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
