# Cluster Evaluation Metrics

> A quick reference guide to internal and external clustering metrics.

## Internal Metrics (No Ground Truth)

*   **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Range: $[-1, 1]$. Higher is better.
*   **Davies-Bouldin Index:** The average similarity measure of each cluster with its most similar cluster. Lower is better.
*   **Calinski-Harabasz Index (Variance Ratio):** Ratio of the sum of between-cluster dispersion to within-cluster dispersion. Higher is better.

## External Metrics (Ground Truth Available)

*   **Adjusted Rand Index (ARI):** Computes a similarity measure between two clusterings. Adjusted for chance. Range: $[-1, 1]$.
*   **Normalized Mutual Information (NMI):** Normalises the Mutual Information score. Range: $[0, 1]$.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
