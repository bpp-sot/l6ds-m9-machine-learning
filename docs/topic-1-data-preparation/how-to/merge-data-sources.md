# How to Merge Multiple Data Sources

> Rarely does all your predictive signal live in a single table. Merging correctly is essential for feature enrichment.

## What You Will Learn
- Execute Left, Right, Inner, and Outer joins
- Merge on discrepant column names 
- Verify merge integrity securely 

## Step 1: The Inner Join

If you want to merge two datasets and strictly only retain the records that securely exist identically in both frames, use the `inner` merge (the Pandas default).

```python
import pandas as pd

# Creating synthetic mock data to simulate relational SQL tables
customers = pd.DataFrame({'cust_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
transactions = pd.DataFrame({'cust_id': [2, 3, 4], 'amount': [150, 200, 50]})

# Only Bob (2) and Charlie (3) exist in both. Alice (1) and (4) are dropped.
inner_merged = pd.merge(customers, transactions, on='cust_id', how='inner')
print(inner_merged)
```

??? example "Expected Output"
    ```text
       cust_id     name  amount
    0        2      Bob     150
    1        3  Charlie     200
    ```

## Step 2: The Left Join (Enrichment)

If predicting customer churn, you absolutely must retain 100% of your primary customers even if they have `0` transactions recorded. Use a `left` join securely to enrich the left-hand DataFrame.

```python
# Retain everyone in 'customers', populate NaN for missing matches in 'transactions'
left_merged = pd.merge(customers, transactions, on='cust_id', how='left')

# Fill NaN transaction amounts with 0 computationally
left_merged['amount'] = left_merged['amount'].fillna(0)
print(left_merged)
```

??? example "Expected Output"
    ```text
       cust_id     name  amount
    0        1    Alice     0.0
    1        2      Bob   150.0
    2        3  Charlie   200.0
    ```

## Step 3: Merging on Discrepant Keys

When databases aren't homogenised, your keys might be named differently (e.g. `cust_id` vs `client_num`). Use `left_on` and `right_on`.

```python
# Simulating a third table with a different ID column completely
address = pd.DataFrame({'client_num': [1, 2, 3], 'city': ['London', 'Manchester', 'Leeds']})

# Merge specifying explicit columns
enriched = pd.merge(left_merged, address, left_on='cust_id', right_on='client_num', how='left')

# Drop the redundant secondary key efficiently
enriched = enriched.drop(columns=['client_num'])
print(enriched)
```

??? example "Expected Output"
    ```text
       cust_id     name  amount        city
    0        1    Alice     0.0      London
    1        2      Bob   150.0  Manchester
    2        3  Charlie   200.0       Leeds
    ```

!!! tip "Workplace Tip"
    Always run `df.shape` before and after executing a `pd.merge()`. Doing a 1:Many join accidentally on a duplicated ID column can massively multiply your row count silently, leading to fatally flawed analytics statistics.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5.3 | Common patterns in real-world data | Identifying missing values, duplicates, outliers, and class imbalance |
| S2 | Data engineering and governance | Systematic data cleaning, transformation, and quality assessment |
| S3 | Programming for data manipulation | pandas pipelines for data preparation |
| B3 | Adaptability and pragmatism | Handling imperfect real-world datasets |
