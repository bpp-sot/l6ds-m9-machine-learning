# Clustering Algorithms Comparison

> A quick reference guide to the most common clustering algorithms.

## Summary

| Algorithm | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **k-Means** | Fast, scalable | Must choose $k$, assumes spherical clusters | Simple baseline, large clean datasets |
| **Hierarchical** | Intuitive dendrogram, no $k$ needed upfront | Slow $O(N^3)$, doesn't scale well | Small datasets, taxonomy building |
| **DBSCAN** | Finds arbitrary shapes, handles noise/outliers | Struggles with varying density, hard to tune eps | Geospatial data, anomaly detection |

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
