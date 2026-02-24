# How to Run Regression Diagnostics

> Regression metrics like \(R^2\) are global summaries. To truly evaluate a continuous model, you must investigate the **residuals** — the errors left behind after prediction.

## What is a Residual?

$$\text{Residual} = \text{True Value} - \text{Predicted Value}$$

If an algorithm predicts a house costs £100,000 but the actual value is £110,000, the residual is +£10,000.

A well-fitted model produces residuals that are **randomly scattered around zero** with no visible pattern.

## Plotting Residuals

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = sns.load_dataset("tips")
X = df[["total_bill"]]
y = df["tip"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_tr, y_tr)
preds = lr.predict(X_te)

residuals = y_te - preds

# 1. Residual scatter plot
plt.figure()
sns.scatterplot(x=preds, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Expected: Random Scatter)")
plt.tight_layout()
plt.show()

# 2. Residual histogram
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (Expected: Normal / Gaussian)")
plt.tight_layout()
plt.show()
```

## Interpreting the Plots

1. **Heteroscedasticity (Fan Shape):** If the residuals widen as predicted values increase, your model's error grows with magnitude. Consider a log transform on the target variable.
2. **Curved Pattern:** A systematic curve in the residuals indicates your model is too simple (high bias). Consider polynomial features or a non-linear algorithm.
3. **Normal Distribution:** Ideally, residuals follow a bell curve centred on zero. Heavy tails or skew suggest outliers are distorting the fit.

!!! tip "Workplace Tip"
    Always plot residuals before reporting \(R^2\). A high \(R^2\) can mask systematic patterns that residual analysis will reveal immediately.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
