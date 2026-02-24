# How to Compare Models Statistically

> Comparing mean CV scores is not enough — you need a statistical test to determine whether the difference between two models is significant.

## The Problem

Model A has a mean CV accuracy of 0.87 and Model B has 0.85. Is A genuinely better, or is the difference just noise from the random fold splits?

## Solution: Paired t-Test on CV Folds

By using the **same cross-validation folds** for both models, each fold produces a paired observation. A paired t-test then determines whether the mean difference is statistically significant.

```python
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from scipy.stats import ttest_rel

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

scores_rf = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=cv, scoring="accuracy")
scores_gb = cross_val_score(GradientBoostingClassifier(random_state=42), X, y, cv=cv, scoring="accuracy")

print(f"RF Mean: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")
print(f"GB Mean: {scores_gb.mean():.4f} ± {scores_gb.std():.4f}")

# Paired t-test
stat, p_value = ttest_rel(scores_rf, scores_gb)
print(f"\nPaired t-test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Significant difference — choose the model with the higher mean.")
else:
    print("No significant difference — models are statistically equivalent.")
```

## Interpretation

| p-value | Conclusion |
|---------|-----------|
| < 0.05 | The difference is statistically significant |
| ≥ 0.05 | No evidence the models differ — prefer the simpler one |

!!! warning "Common Pitfall"
    You must use the **same CV folds** for both models. Using different random splits invalidates the paired comparison.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
