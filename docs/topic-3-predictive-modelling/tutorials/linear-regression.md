# Linear Regression under the Hood

> "All models are wrong, but some are useful." — George Box

## What You Will Learn

- Understand the mathematical foundation of Ordinary Least Squares (OLS)
- Train a `LinearRegression` model using Scikit-Learn
- Interpret coefficients ($\beta$) and the intercept ($\beta_0$)
- Evaluate a model using $R^2$ and Mean Squared Error (MSE)

## Prerequisites

- [Pipelines in Data Prep](../../topic-1-data-preparation/tutorials/pipelines.md)

## Step 1: The Equation of a Line

Linear regression attempts to fit the "best" straight line through a scatterplot of data points. For a simple regression (one feature), the equation is:

\\[
y = \beta_0 + \beta_1 x + \epsilon
\\]

Where:
- $y$ is the predicted target (e.g., House Price)
- $x$ is the input feature (e.g., Square Footage)
- $\beta_0$ is the y-intercept (the predicted price if square footage was exactly 0)
- $\beta_1$ is the coefficient (the slope—how much the price increases for every 1 unit increase in $x$)
- $\epsilon$ is the error term (residuals)

If you have multiple features (Multiple Linear Regression), it expands infinitely:
\\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n
\\]

## Step 2: Training the Model (Ordinary Least Squares)

The algorithm calculates the "best" line by minimizing the sum of the squared distances (residuals) between the actual data points and the predicted line on the graph. This is called **Ordinary Least Squares (OLS)**.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic housing data
np.random.seed(42)
sqft = np.random.normal(1500, 500, 200)

# True relationship: Base price 50k + $200 per sqft + Noise
price = 50000 + (200 * sqft) + np.random.normal(0, 30000, 200) 

df = pd.DataFrame({'SqFt': sqft, 'Price': price})

# Scikit-learn requires 2D arrays for features
X = df[['SqFt']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Instantiate the Model
model = LinearRegression()

# 2. Fit the Model (This runs OLS under the hood)
model.fit(X_train, y_train)

print(f"Intercept (Beta 0): £{model.intercept_:,.2f}")
print(f"Coefficient (Beta 1): £{model.coef_[0]:,.2f} per SqFt")
```

## Step 3: Evaluation Metrics

Unlike Classification models (which use Accuracy), Regression models predict a continuous number. 

1. **Mean Squared Error (MSE) / Root Mean Squared Error (RMSE):** The average squared distance between predictions and actuals. RMSE puts the error back into the original units (e.g., Pounds).
2. **R-Squared ($R^2$):** The proportion of variance in the target variable that is predictable from the features. An $R^2$ of 1.0 is perfect; 0.0 means the model is as good as just guessing the historical average.

```python
# 3. Predict on Test Set
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"\\nRMSE: £{rmse:,.2f}")
print(f"R-Squared: {r2:.4f}")

# Visualizing the Fit
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['SqFt'], y=y_test, color='blue', label='Actual Test Data')
plt.plot(X_test['SqFt'], predictions, color='red', linewidth=3, label='OLS Regression Line')
plt.title('Linear Regression: SqFt vs Price')
plt.xlabel('Square Footage')
plt.ylabel('Price (£)')
plt.legend()
plt.show()
```

## Summary

Linear Regression is the grandfather of all machine learning algorithms. Its primary strength lies in **Interpretability**. You can mathematically explain exactly why a prediction was made by looking at the $\beta$ coefficients. In highly regulated sectors (Banking, Healthcare), Linear/Logistic Regression is often mandatory.

## Next Steps

→ [Logistic Regression for Classification](logistic-regression.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S2 | Apply mathematical algorithms | Building algebraic parameters using Least Squares estimators |
| K2 | Machine learning paradigms | Implementing Regression structures and evaluation metrics |
