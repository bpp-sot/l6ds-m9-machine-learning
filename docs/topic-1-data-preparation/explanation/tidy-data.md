# Tidy Data Principles

> Tidy Data is the fundamental geometric architecture required for algorithmic ingestion.

## The Three Rules of Tidy Data

Before any algorithm can ingest your DataFrame, it must structurally adhere rigidly to the "Tidy Data" principles popularized by Hadley Wickham.

1. **Each variable must have its own column.**
2. **Each observation must have its own row.**
3. **Each value must have its own cell.**

## Example: Untidy Data (Wide Format)

Often, stakeholders will send you data designed for human readability (like a pivot table), which is fundamentally broken for computational learning.

| Region | 2021_Sales | 2022_Sales | 2023_Sales |
|---|---|---|---|
| London | 1500 | 1700 | 2000 |
| Manchester | 800 | 850 | 900 |

**Why is this broken?**
- `2021_Sales` and `2022_Sales` are not separate variables; they are the exact same variable (`Sales`) recorded at different chronological times. 
- The actual variable `Year` is hiding dynamically inside the column headers!

## The Solution: Tidy Data (Long Format)

We must use the Pandas `.melt()` function to reshape the geometry of this table.

```python
import pandas as pd

untidy_df = pd.DataFrame({
    'Region': ['London', 'Manchester'],
    '2021_Sales': [1500, 800],
    '2022_Sales': [1700, 850]
})

# Melt the DataFrame structurally
tidy_df = untidy_df.melt(
    id_vars=['Region'], 
    var_name='Year', 
    value_name='Sales'
)

# Clean up the Year string vectorised
tidy_df['Year'] = tidy_df['Year'].str.replace('_Sales', '').astype(int)
print(tidy_df)
```

??? example "Expected Output"
    ```text
           Region  Year  Sales
    0      London  2021   1500
    1  Manchester  2021    800
    2      London  2022   1700
    3  Manchester  2022    850
    ```

Now, your data cleanly adheres to the rule: 1 Observation = 1 Row. You can successfully feed this matrix natively into an algorithm to predict `Sales` using `Region` and `Year` as independent predictive coordinates.

!!! tip "Workplace Tip"
    Whenever a colleague sends you a formatted Excel sheet with merged cells, multiple header rows, and colour-coded highlights, you must immediately strip all visual formatting and `.melt()` the data into a strict flat Tidy format before attempting any Pythonic analysis.

## KSB Mapping

| KSB | Description | How This Explanation Addresses It |
|-----|-------------|-------------------------------|
| K2 | Internal and External data structures | Identifying the structural geometric constraints of tabular schemas |
| S4 | Import, cleanse, transform data | Flattening matrix arrays utilizing logical pivoting algorithms |
