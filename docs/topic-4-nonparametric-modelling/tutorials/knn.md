# k-Nearest Neighbours (k-NN)

> k-NN is a simple, instance-based algorithm: to classify a new observation, it finds the \(k\) closest training points and takes a majority vote.

## How It Works

1. Store the entire training dataset in memory (no explicit "training" step).
2. When a new observation arrives, calculate its distance to every training point.
3. Select the \(k\) nearest neighbours.
4. Return the majority class (classification) or the mean value (regression).

There are no learned parameters — the model *is* the data. This makes k-NN a **lazy learner**.

## Implementation

```python
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("penguins").dropna()
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]]
y = df["species"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# CRITICAL: k-NN is distance-based — you MUST standardise features
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr_sc, y_tr)
print(classification_report(y_te, knn.predict(X_te_sc)))
```

## Key Considerations

| Consideration | Detail |
|---------------|--------|
| **Feature scaling** | Mandatory — features on larger scales dominate the distance calculation |
| **Curse of dimensionality** | Performance degrades rapidly as the number of features grows |
| **Prediction speed** | Slow on large datasets (must compute distance to every training point) |
| **No feature importance** | k-NN provides no insight into which features drive predictions |

## Choosing \(k\)

- **Small \(k\)** (e.g., 1–3): Highly sensitive to noise, risk of overfitting.
- **Large \(k\)** (e.g., 50+): Over-smoothed boundaries, risk of underfitting.
- Use cross-validation to find the optimal value (see [How to Choose k](../how-to/choose-k-value.md)).

!!! warning "Common Pitfall"
    Forgetting to standardise features before fitting k-NN is the single most common mistake. Without scaling, a feature measured in thousands (e.g., salary) will completely dominate one measured in decimals (e.g., GPA).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced ML techniques | Tree-based models, ensemble methods, KNN, SVM |
| K4.4 | Trade-offs in selecting algorithms | Comparing parametric vs non-parametric approaches |
| S4 | ML and optimisation | Hyperparameter tuning, ensemble construction, model selection |
| B1 | Curiosity and creativity | Exploring when non-parametric methods outperform parametric ones |
| B5 | Integrity in presenting conclusions | Avoiding overfitting; honest reporting of generalisation performance |
