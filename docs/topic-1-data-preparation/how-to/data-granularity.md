# How to Change Data Granularity

> Machine Learning algorithms require one row per 'event'. If your data is recorded per minute, but your predictive event is 'Total Daily Sales', you must aggregate granularities.

## What You Will Learn
- Downsample high-frequency time-series data using `.resample()`
- Aggregate categorical transactions into grouped numeric rows using `.groupby()`

## Step 1: Time-Series Downsampling

If sensors log temperature every 5 seconds, you will have $17,280$ rows per day. If you only want to predict the "Daily High Temp", you must compress the chronological granularity computationally.

```python
import pandas as pd
import numpy as np

# Simulate 1 week of minute-by-minute sensor data
dates = pd.date_range("2024-01-01", "2024-01-07", freq="min")
df = pd.DataFrame({
    'timestamp': dates,
    'temperature_c': np.random.normal(20, 2, len(dates))
})

# Set the timestamp as the index (mandatory for Pandas time-series)
df.set_index('timestamp', inplace=True)

# Resample to Daily ('D') frequency, capturing the Max and Mean
daily_stats = df.resample('D').agg({
    'temperature_c': ['max', 'mean']
})

print(daily_stats)
```

??? example "Expected Output"
    ```text
                temperature_c           
                          max       mean
    timestamp                           
    2024-01-01      25.105123  20.001235
    2024-01-02      26.002341  19.981290
    ...
    ```

## Step 2: Aggregating Categorical Groups

If a transactional file has 1 row per checkout receipt, but your objective is to predict "Customer Lifetime Value", you must aggregate the data so the granularity is exactly 1 row per `customer_id`.

```python
# Simulating receipt-level transactions
transactions = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2, 3],
    'purchase_amount': [15.50, 20.00, 5.00, 100.00, 50.00, 12.00],
    'item_category': ['Food', 'Food', 'Drink', 'Electronics', 'Clothing', 'Food']
})

# Group statically by Customer
customer_profile = transactions.groupby('customer_id').agg(
    total_spent=('purchase_amount', 'sum'),
    average_order_value=('purchase_amount', 'mean'),
    number_of_purchases=('purchase_amount', 'count')
).reset_index()

print(customer_profile)
```

??? example "Expected Output"
    ```text
       customer_id  total_spent  average_order_value  number_of_purchases
    0            1         40.5                13.50                    3
    1            2        150.0                75.00                    2
    2            3         12.0                12.00                    1
    ```

!!! info "Assessment Connection"
    In your presentation, examiners will probe whether your dataset granularity matched your theoretical problem statement. Demonstrating explicit `.groupby()` engineering proves you didn't simply throw raw transactional logs blindly into an algorithm!

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5.3 | Common patterns in real-world data | Identifying missing values, duplicates, outliers, and class imbalance |
| S2 | Data engineering and governance | Systematic data cleaning, transformation, and quality assessment |
| S3 | Programming for data manipulation | pandas pipelines for data preparation |
| B3 | Adaptability and pragmatism | Handling imperfect real-world datasets |
