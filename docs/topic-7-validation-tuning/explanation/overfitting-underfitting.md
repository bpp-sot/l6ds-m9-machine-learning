# Overfitting vs Underfitting

> The primary goal of machine learning is to find the sweet spot between bias (underfitting) and variance (overfitting).

## Underfitting (High Bias)
The model is too simple to capture the underlying pattern of the data.
*   **Symptom:** Bad performance on both the training set and the test set.
*   **Cure:** Use a more complex model (e.g., switch from Linear Regression to Random Forest), add more features, or reduce regularisation.

## Overfitting (High Variance)
The model is too complex and has memorised the training data, including its noise and outliers.
*   **Symptom:** Excellent performance on the training set, but terrible performance on the test set.
*   **Cure:** Get more data, use a simpler model, use regularisation, or use early stopping.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.4 | Resource constraints and trade-offs | Balancing model complexity, performance, and computational cost |
| S1 | Scientific methods and hypothesis testing | Rigorous cross-validation and statistical model comparison |
| S4 | Building models and validating | Systematic hyperparameter tuning and performance evaluation |
| B5 | Impartial, hypothesis-driven approach | Preventing overfitting; honest reporting of generalisation metrics |
