# Clustering Algorithms Comparison

> A quick reference guide to the most common clustering algorithms.

## Summary

| Algorithm | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **k-Means** | Fast, scalable | Must choose $k$, assumes spherical clusters | Simple baseline, large clean datasets |
| **Hierarchical** | Intuitive dendrogram, no $k$ needed upfront | Slow $O(N^3)$, doesn't scale well | Small datasets, taxonomy building |
| **DBSCAN** | Finds arbitrary shapes, handles noise/outliers | Struggles with varying density, hard to tune eps | Geospatial data, anomaly detection |

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
