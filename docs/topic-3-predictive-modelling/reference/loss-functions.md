# Common Loss Functions

> A loss function (or cost function) is the mathematical equation an algorithm uses to calculate how "wrong" its predictions are. The algorithm then adjusts its parameters to minimise this value.

## Regression Loss Functions

Used by algorithms predicting continuous values (e.g., `LinearRegression`, `Ridge`).

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Heavily punishes large errors because the difference is squared.
- Sensitive to outliers — a single extreme prediction dominates the total.
- **Use when:** Large errors are disproportionately costly (e.g., financial forecasting).

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Treats all errors equally regardless of magnitude.
- More robust to outliers than MSE.
- **Use when:** You want a metric that reflects typical error magnitude without outlier distortion.

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Same units as the target variable, making it easier to interpret than MSE.
- Still penalises large errors heavily due to the underlying squaring.

## Classification Loss Functions

Used by algorithms predicting discrete categories (e.g., `LogisticRegression`, `RandomForestClassifier`).

### Log Loss (Binary Cross-Entropy)

$$\text{Log Loss} = -\frac{1}{n} \sum [y \log(\hat{p}) + (1 - y) \log(1 - \hat{p})]$$

- Punishes confident but incorrect predictions severely (e.g., predicting 99% probability of Class 1 when the true label is Class 0).
- **Use when:** You need well-calibrated probability outputs, not just hard class labels.

### Hinge Loss

$$\text{Hinge} = \max(0, 1 - y \cdot \hat{y})$$

- Used by Support Vector Machines (SVM).
- Penalises misclassifications and observations that fall within the margin, but incurs zero loss for correctly classified points far from the boundary.

## Quick Reference Table

| Loss Function | Problem Type | Outlier Sensitivity | scikit-learn Parameter |
|---------------|-------------|--------------------|-----------------------|
| MSE | Regression | High | `scoring="neg_mean_squared_error"` |
| MAE | Regression | Low | `scoring="neg_mean_absolute_error"` |
| Log Loss | Classification | Medium | `scoring="neg_log_loss"` |
| Hinge | Classification | Low | `scoring="hinge"` (SVM) |

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
