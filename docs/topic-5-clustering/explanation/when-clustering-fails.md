# When Clustering Fails

> Clustering algorithms will almost always return *something*. Knowing when those results are meaningless is a critical data science skill.

## Common Failure Modes

1.  **Forcing Structure where None Exists:** Data might just be one big blob naturally. k-Means will still cut it into pieces.
2.  **The Curse of Dimensionality:** In very high-dimensional space, all points look equidistant from each other. Distance metrics break down.
3.  **Varying Densities:** DBSCAN struggles if cluster A is very dense but cluster B is very sparse.
4.  **Non-Spherical Data for k-Means:** k-Means functionally draws circles (or spheres/hyper-spheres). If your data is shaped like concentric rings or bananas, k-Means will fail dramatically.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
