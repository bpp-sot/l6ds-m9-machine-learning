# Choosing the Right Metric

> You can have a model with 99% accuracy that is completely useless.

## The Imbalanced Data Trap
Imagine a dataset predicting credit card fraud. 99% of transactions are legitimate, and 1% are fraudulent.
A "dumb" model that simply predicts "Legitimate" for every single transaction will score 99% Accuracy. But it caught 0 frauds.

## When to use what:
*   **Accuracy:** Only when classes are perfectly balanced (e.g., 50% cats, 50% dogs).
*   **Precision:** When **false positives are expensive**. (e.g., A spam filter. You don't want to send legitimate emails to the junk folder).
*   **Recall:** When **false negatives are expensive**. (e.g., Cancer screening. Missing a cancer diagnosis is worse than a false alarm).
*   **F1-Score:** When you want a balance of Precision and Recall on an imbalanced dataset.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
