# Clustering vs Classification

> Both group data, but they operate under fundamentally different paradigms.

## The Core Difference

*   **Classification (Supervised):** You know the answers (labels). You train the algorithm to map inputs to those known answers. The goal is prediction.
*   **Clustering (Unsupervised):** You do *not* know the answers. You ask the algorithm to find natural groupings in the raw data. The goal is exploration and discovery.

## When to use which?

*   Use classification if you have a historically labelled dataset (e.g., predicting if a *known* customer churned).
*   Use clustering if you have raw data and want to understand it (e.g., finding *groups* of similar customers to create targeted marketing campaigns without knowing what those groups are beforehand).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
