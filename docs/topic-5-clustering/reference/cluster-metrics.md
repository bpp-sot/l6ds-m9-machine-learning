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

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
