# Build a Customer Segmentation

> Apply k-Means to RFM (Recency, Frequency, Monetary) features to group customers into actionable segments.

## What Is RFM?

RFM analysis scores each customer on three dimensions:

| Metric | Meaning |
|--------|---------|
| **Recency** | How recently did they purchase? (lower = better) |
| **Frequency** | How often do they purchase? (higher = better) |
| **Monetary** | How much do they spend in total? (higher = better) |

## Implementation

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulate RFM data
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "recency": rng.integers(1, 365, size=200),
    "frequency": rng.integers(1, 50, size=200),
    "monetary": rng.integers(10, 5000, size=200)
})

# CRITICAL: Scale features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal k using inertia
inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto")
            .fit(X_scaled).inertia_ for k in range(2, 9)]

plt.plot(range(2, 9), inertias, "bo-")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method — Customer Segments")
plt.show()

# Apply final clustering
km = KMeans(n_clusters=4, random_state=42, n_init="auto")
df["segment"] = km.fit_predict(X_scaled)

# Inspect segment profiles
print(df.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1))
```

## Interpreting Segments

After clustering, label each segment based on its RFM profile:

| Segment | Recency | Frequency | Monetary | Label |
|---------|---------|-----------|----------|-------|
| 0 | Low | High | High | Champions |
| 1 | High | Low | Low | At Risk |
| 2 | Medium | Medium | Medium | Loyal |
| 3 | Low | Low | Low | New Customers |

!!! tip "Workplace Tip"
    Always standardise your RFM features before clustering. Without scaling, monetary values (in thousands) will dominate recency (in days) and frequency (in single digits).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Unsupervised learning algorithms for pattern discovery |
| K4.4 | Trade-offs in selecting algorithms | Choosing between clustering approaches based on data characteristics |
| S1 | Scientific methods and hypothesis testing | Validating cluster quality without ground truth labels |
| S4 | Analysis and models to inform outcomes | Using clustering to derive actionable segments |
| B1 | Inquisitive approach | Exploring hidden structure in unlabelled data |
