# Naive Bayes

> A fast, scalable classification algorithm based on Bayes' Theorem with a "naive" assumption that all features are conditionally independent given the class.

## The Concept

Naive Bayes calculates the probability of each class given the observed feature values using Bayes' Theorem:

$$P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}$$

The "naive" assumption is that each feature contributes independently to the probability. This is almost never true in reality, yet Naive Bayes still performs surprisingly well — especially on text classification and high-dimensional sparse data.

## Variants

| Variant | Use Case | Assumption |
|---------|----------|------------|
| `GaussianNB` | Continuous features | Features follow a normal distribution per class |
| `MultinomialNB` | Count data (e.g., word frequencies) | Features represent counts or frequencies |
| `BernoulliNB` | Binary features (0/1) | Features are binary indicators |

## Implementation

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

nb = GaussianNB()
nb.fit(X_tr, y_tr)
print(classification_report(y_te, nb.predict(X_te)))
```

## Strengths and Weaknesses

**Strengths:**

- Extremely fast to train and predict — scales linearly with dataset size.
- Works well with high-dimensional data (e.g., text with thousands of word features).
- Handles missing data gracefully (each feature probability is computed independently).

**Weaknesses:**

- The independence assumption means it cannot capture feature interactions.
- Probability estimates are often poorly calibrated (overconfident).
- Outperformed by tree-based ensembles on most tabular datasets.

!!! tip "Workplace Tip"
    Use Naive Bayes as a **fast baseline**. If it achieves 85% accuracy in seconds, you know any more complex model should beat that. It is also the go-to choice for spam filtering and text classification.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Implementing a probabilistic classifier and understanding its assumptions |
