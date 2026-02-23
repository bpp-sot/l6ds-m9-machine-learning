# How-to: Merge Multiple Data Sources

## The Problem
Workplaces don't store perfect analytic sets in a single CSV. Transaction data lives in one database, customer profiles in Salesforce, and operational logs in an Excel spreadsheet. You must join them.

## The Solution
Pandas offers `merge()`, which functions identically to a SQL JOIN.

```python
import pandas as pd

# Creating sample datasets
customers = pd.DataFrame({
    'CustomerID': [101, 102, 103, 104],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Segment': ['Retail', 'Corporate', 'Retail', 'Corporate']
})

transactions = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4],
    'CustomerID': [101, 101, 105, 102], # Notice 105 doesn't exist in customers!
    'Amount': [250, 150, 400, 300]
})

# Inner Join (Default)
# only keeps customers who have BOTH a profile and a transaction
inner_merged = pd.merge(customers, transactions, on='CustomerID', how='inner')
print("Inner Join (Intersection):")
print(inner_merged)

# Left Join
# keeps ALL customers, even if they have no transactions (NaN amount)
left_merged = pd.merge(customers, transactions, on='CustomerID', how='left')
print("\\nLeft Join (All Customers):")
print(left_merged)
```

## Discussion

### When to use `how='left'` vs `how='inner'`?
- Use left joins when the left table is your "master list" (e.g., you are building a predictive model for *all* registered clients, regardless of whether they have purchased recently).
- Use inner joins when you intend to drop records with incomplete cross-table alignment.

### Common Pitfalls
- **Duplicating Rows (The Cartesian Explosion)**: If your `CustomerID` isn't unique in the secondary table, `merge` generates identical rows. Always run `.duplicated('CustomerID').sum()` before a merge.
- **Suffix Clashes**: If both tables have a `'Date'` column not used in the `on=` parameter, Pandas creates `'Date_x'` and `'Date_y'`. Handle this cleanly via `pd.merge(..., suffixes=('_cust', '_trans'))`.
