# Advanced Tree-Based Methods

> Beyond standard Random Forests lie specialised architectures designed for speed, isolation detection, or extreme randomisation.

## ExtraTrees (Extremely Randomised Trees)

A standard Random Forest searches for the **best** split at each node. ExtraTrees instead picks split thresholds **at random**, which is faster and often reduces variance further.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

et = ExtraTreesClassifier(n_estimators=100, random_state=42)
score = cross_val_score(et, X, y, cv=5, scoring="accuracy").mean()
print(f"ExtraTrees CV Accuracy: {score:.3f}")
```

**When to use:** When you want faster training than Random Forest with comparable (sometimes better) accuracy.

## Isolation Forest (Anomaly Detection)

Isolation Forest detects outliers by building random trees and measuring how quickly each observation is isolated. Anomalies are isolated in fewer splits because they sit in sparse regions of the feature space.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Generate data with outliers
rng = np.random.default_rng(42)
X_normal = rng.normal(size=(200, 2))
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
X_all = np.vstack([X_normal, X_outliers])

iso = IsolationForest(contamination=0.1, random_state=42)
labels = iso.fit_predict(X_all)  # -1 = outlier, 1 = inlier
print(f"Outliers detected: {(labels == -1).sum()}")
```

**When to use:** Fraud detection, sensor anomaly detection, or any scenario where you need to flag unusual observations without labelled data.

## Comparison Table

| Algorithm | Split Strategy | Primary Use | Speed |
|-----------|---------------|-------------|-------|
| `RandomForestClassifier` | Best split from random feature subset | General classification/regression | Medium |
| `ExtraTreesClassifier` | Random split thresholds | Same as RF, faster training | Fast |
| `IsolationForest` | Random isolation depth | Anomaly / outlier detection | Fast |

!!! info "Assessment Connection"
    Demonstrating awareness of specialised tree variants (not just Random Forest) shows breadth of algorithmic knowledge for your EPA.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced ML techniques | Tree-based models, ensemble methods, KNN, SVM |
| K4.4 | Trade-offs in selecting algorithms | Comparing parametric vs non-parametric approaches |
| S4 | ML and optimisation | Hyperparameter tuning, ensemble construction, model selection |
| B1 | Curiosity and creativity | Exploring when non-parametric methods outperform parametric ones |
| B5 | Integrity in presenting conclusions | Avoiding overfitting; honest reporting of generalisation performance |
