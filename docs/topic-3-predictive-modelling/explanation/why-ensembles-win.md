# Why Ensembles Win

> In modern Data Science, a standalone Decision Tree or Logistic Regression is almost never deployed to production on its own. Ensembles dominate because they reduce error by combining multiple models.

## The Problem with Single Models

Every individual algorithm has a weakness:

| Model | Weakness |
|-------|----------|
| Linear Regression | Cannot capture non-linear relationships |
| Decision Tree | Extremely high variance — small data changes produce wildly different trees |
| KNN | Collapses in high-dimensional spaces (curse of dimensionality) |

No single model can be simultaneously low-bias *and* low-variance. Ensembles solve this by **aggregating** multiple models so their individual weaknesses cancel out.

## Why Aggregation Works

Consider a binary classification task. If you train 100 independent models, each with 60% accuracy (barely better than random), and let them **vote**:

- The probability that the majority vote is correct rises dramatically as the number of independent voters increases.
- This is a direct application of the **Law of Large Numbers**.

The critical requirement is **diversity** — if all 100 models make the same mistakes, voting achieves nothing. Bagging injects diversity via random bootstrap samples; boosting injects it by focusing each successive model on previously misclassified observations.

## Real-World Dominance

- **Kaggle competitions:** The vast majority of winning solutions on tabular data use XGBoost, LightGBM, or stacked ensembles.
- **Industry deployments:** Gradient Boosting is the default choice for fraud detection, churn prediction, credit scoring, and recommendation systems.
- **scikit-learn defaults:** Even `RandomForestClassifier` with default parameters frequently outperforms a tuned single model.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

for name, model in [("LogReg", LogisticRegression(max_iter=1000)),
                     ("RF", RandomForestClassifier(random_state=42)),
                     ("GB", GradientBoostingClassifier(random_state=42))]:
    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    print(f"{name}: {score:.3f}")
```

!!! info "Assessment Connection"
    In your EPA, justify *why* you chose an ensemble over a simpler model. Stating that ensembles reduce variance through aggregation demonstrates understanding of S13 — Apply ML Algorithms.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
