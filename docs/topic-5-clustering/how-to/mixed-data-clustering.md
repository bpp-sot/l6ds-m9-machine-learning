# Cluster Mixed Data Types

> When data has both numerical and categorical columns, standard distance metrics (Euclidean) break down. Gower's Distance handles both types in a single calculation.

## The Problem

k-Means uses Euclidean distance, which is undefined for categorical features. You cannot simply label-encode categories and treat them as numbers — the numeric distances between arbitrary codes are meaningless.

## Gower's Distance

Gower's Distance computes a normalised dissimilarity for each feature pair based on its type:

- **Numerical:** Range-normalised absolute difference.
- **Categorical:** Binary (0 if same category, 1 if different).

The overall distance is the weighted average across all features.

```python
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# pip install gower
import gower

df = pd.DataFrame({
    "age": [25, 35, 45, 30, 50],
    "income": [30000, 50000, 70000, 40000, 80000],
    "region": ["North", "South", "North", "South", "North"],
    "membership": ["Gold", "Silver", "Gold", "Bronze", "Gold"]
})

# Compute Gower distance matrix
dist_matrix = gower.gower_matrix(df)
print(f"Distance matrix shape: {dist_matrix.shape}")

# Cluster using the precomputed distance matrix
model = AgglomerativeClustering(
    n_clusters=2,
    metric="precomputed",
    linkage="average"
)
df["cluster"] = model.fit_predict(dist_matrix)
print(df)
```

!!! tip "Workplace Tip"
    For large mixed-type datasets, consider `k-Prototypes` from the `kmodes` library, which extends k-Means to handle mixed data directly without computing a full distance matrix.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
