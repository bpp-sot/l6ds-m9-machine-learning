# Datasets

> A curated list of built-in datasets used throughout this module. No downloads or API keys required.

## Seaborn Datasets

Load any of these with `sns.load_dataset('name')`:

| Dataset | Task | Target Variable |
|---------|------|-----------------|
| `titanic` | Binary Classification | `survived` (0/1) |
| `penguins` | Multiclass Classification | `species` (Adelie, Chinstrap, Gentoo) |
| `tips` | Regression | `tip` (continuous) |
| `taxis` | Regression | `fare` (continuous) |
| `diamonds` | Regression | `price` (continuous) |
| `iris` | Multiclass Classification | `species` |

```python
import seaborn as sns

df = sns.load_dataset("titanic")
print(df.shape)
print(df.head())
```

## Scikit-Learn Datasets

Load these with `from sklearn.datasets import <function>`:

| Function | Task | Samples | Features |
|----------|------|---------|----------|
| `load_breast_cancer()` | Binary Classification | 569 | 30 |
| `load_iris()` | Multiclass Classification | 150 | 4 |
| `load_diabetes()` | Regression | 442 | 10 |
| `load_wine()` | Multiclass Classification | 178 | 13 |

### Synthetic Generators

| Function | Purpose |
|----------|---------|
| `make_classification()` | Generate synthetic classification data with controllable complexity |
| `make_regression()` | Generate synthetic regression data |
| `make_blobs()` | Generate clustered data for unsupervised learning |
| `make_moons()` | Generate non-linear, crescent-shaped clusters |

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
```

## KSB Mapping

| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
