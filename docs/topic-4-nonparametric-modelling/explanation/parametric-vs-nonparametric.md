# Parametric vs. Non-Parametric Models

> Understanding the fundamental difference in how models assume data structure.

## Parametric Models
Parametric models (like Linear Regression or Logistic Regression) assume a specific functional form for the relationship between inputs and outputs.

*   **Fixed Parameters:** They have a fixed number of parameters, regardless of the training data size.
*   **Pros:** Fast to train, fast to predict, require less data, highly interpretable.
*   **Cons:** Can be overly simplistic, prone to underfitting if the true relationship is complex (high bias).

## Non-Parametric Models
Non-parametric models (like k-NN, Decision Trees, or SVM with RBF kernel) do not make strong assumptions about the form of the mapping function.

*   **Flexible Parameters:** The number of parameters (or complexity) grows with the size of the training data.
*   **Pros:** Highly flexible, can model complex non-linear relationships, better fit to the training data.
*   **Cons:** Slower to predict (e.g., k-NN), require more data, prone to overfitting (high variance).

## Summary
Parametric models are rigid but fast and simple. Non-parametric models are flexible but computationally heavy and prone to memorising noise.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
