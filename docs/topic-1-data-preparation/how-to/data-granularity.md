# How-to: Choose the Right Data Granularity

## The Problem
Your machine learning model will fail if your rows do not align logically with the target you are trying to predict. If you are predicting "Daily Revenue", inputting hourly transaction rows makes no sense.

## The Solution
You must aggregate or unpack your dataset to achieve the correct granularity (the definition of a single row). 

```python
import pandas as pd
import numpy as np

# Sample Transaction-level data (Granularity: 1 row = 1 receipt)
transactions = pd.DataFrame({
    'TrxDate': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
    'CustomerID': [101, 101, 102, 101],
    'Amount': [50.50, 100.00, 20.00, 30.00],
    'ProductID': ['A', 'B', 'A', 'C']
})

print("Original Transaction Data:")
print(transactions)

# Goal: Predict "total future customer value". 
# Solution: Group to Customer Granularity (1 row = 1 Customer)
customer_level = transactions.groupby('CustomerID').agg({
    'Amount': ['sum', 'mean', 'count'], # Total Spent, Avg Order Value, Order Count
    'TrxDate': ['min', 'max']           # First Purchase, Recent Purchase
})

# Flatten MultiIndex columns created by agg
customer_level.columns = ['_'.join(col).strip() for col in customer_level.columns.values]
customer_level.reset_index(inplace=True)

print("\\nRe-aggregated Customer Data:")
print(customer_level)
```

## Discussion

### Granularity Mismatches 
A common failure in ML happens when merging varying granularities. Joining city-level demographic data (1 row = 1 City) onto user purchases (1 row = 1 receipt) replicates the city metrics 5,000 times, introducing artificial certainty to the model constraints.

### Temporal Granularity
Time-series forecasting is particularly vulnerable. Ensure you resample (using `df.resample('D').sum()`) correctly to establish explicit frequencies before feeding parameters into an ARIMA or Prophet pipeline.
