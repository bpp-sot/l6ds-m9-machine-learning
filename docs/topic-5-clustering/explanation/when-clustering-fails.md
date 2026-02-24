# When Clustering Fails

> Clustering algorithms will almost always return *something*. Knowing when those results are meaningless is a critical data science skill.

## Common Failure Modes

1.  **Forcing Structure where None Exists:** Data might just be one big blob naturally. k-Means will still cut it into pieces.
2.  **The Curse of Dimensionality:** In very high-dimensional space, all points look equidistant from each other. Distance metrics break down.
3.  **Varying Densities:** DBSCAN struggles if cluster A is very dense but cluster B is very sparse.
4.  **Non-Spherical Data for k-Means:** k-Means functionally draws circles (or spheres/hyper-spheres). If your data is shaped like concentric rings or bananas, k-Means will fail dramatically.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
