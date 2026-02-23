# How-to: Reduce Multicollinearity

## The Problem
**Multicollinearity** occurs when two or more independent variable features are highly correlated with each other in a regression model. 

For example, if you include both `Year_of_Birth` and `Age_in_Years` in a Linear Regression to predict `Salary`, the model will mathematically panic. The coefficients will violently swing, making interpretation completely impossible.

## The Solution
We must compute the **Variance Inflation Factor (VIF)**. Any feature with a VIF score $> 10$ (or sometimes $> 5$) is too collinear and must be removed.

```python
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Create a highly collinear dataset
df = pd.DataFrame({
    'Beds': [2, 3, 4, 3, 5],
    'Baths': [2, 2, 3, 2, 4],
    'SquareFeet': [1200, 1500, 2200, 1600, 3000],
    'SquareMeters': [111, 139, 204, 148, 278] # This is literally SqFt / 10.764
})

# 1. Function to iteratively calculate and drop high VIF features
def calculate_vif(data_frame):
    # Statsmodels requires a constant (intercept) to be added for correct VIF logic
    X = add_constant(data_frame)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data.sort_values('VIF', ascending=False)

print("Initial VIF Scores:")
print(calculate_vif(df))

# 2. Drop the redundant feature and re-calculate
print("\\nVIF Scores after dropping 'SquareMeters':")
df_dropped = df.drop('SquareMeters', axis=1)
print(calculate_vif(df_dropped))
```

## Discussion

### When to use this approach?
You must rigorously check VIF before running **Linear Regression** or **Logistic Regression**. 

### When to ignore this approach?
Tree-based algorithms (Random Forest, XGBoost) and non-parametric systems (KNN, SVM) are functionally immune to multicollinearity. They will simply select one of the correlated columns during a split and ignore the other. If you are exclusively using Trees, VIF calculation is a waste of time.
