# Reference: Pandas Cheatsheet

This page provides quick lookup syntax for Data Preparation and Data Munging functions specifically utilizing Pandas structure variables. 

## Data Loading and Initial Audit

| Task | Syntax | Example |
|------|--------|---------|
| Read CSV | `pd.read_csv()` | `pd.read_csv('data.csv', index_col=0)` |
| Overview | `df.info()` | Prints Non-Null Count & Type mapping |
| Summary Stats| `df.describe()` | Output: count, mean, std, min, 25%, 50%, 75%, max |
| Unique Values | `df['col'].nunique()` | Count distinct values in a column |
| Value Counts | `df['col'].value_counts()`| Frequency counts of distinct values |

## Missing Data Operations

| Task | Syntax | Action |
|------|--------|--------|
| Detect missing | `df.isnull().sum()` | Total NaNs per column |
| Drop exactly | `df.dropna(subset=['A'])` | Remove Rows where 'A' is missing |
| Fill constant | `df.fillna(0)` | Replaces every NaN with '0' |
| Fill logical | `df['A'].fillna(df['A'].median())` | Computes column median and overwrites NaNs |

## Filtering and Selection

```python
import pandas as pd

# 1. Filter rows by condition (boolean indexing)
high_salary = df[df['Salary'] > 60000]

# 2. Filter rows using multiple conditions (AND = &, OR = |)
# Parentheses are strictly REQUIRED around each condition!
target_group = df[(df['Age'] > 30) & (df['City'] == 'London')]

# 3. Locate explicit row and column intersections (Label-based)
value = df.loc[10:20, ['Age', 'Salary']]

# 4. Locate via Integer index (Position-based)
rows = df.iloc[0:5, 0:2] # First 5 rows, first 2 columns
```

## Transformation & Construction

| Task | Syntax | Action |
|------|--------|--------|
| Create Column | `df['New'] = df['A'] + df['B']`| Appends computed feature |
| Map Values | `df['B'] = df['A'].map({'Yes':1, 'No':0})` | Dictionary translation for categories |
| Aggregate | `df.groupby('A').agg({'B': 'sum'})`| Roll up rows logically | 
| Merge | `pd.merge(df1, df2, on='ID')`| SQL-style joining |
| Concatenate | `pd.concat([df1, df2], axis=0)`| Stack DataFrames vertically |

## Common Export

```python
# Write to CSV, dropping the auto-generated DataFrame index
df.to_csv('cleaned_data.csv', index=False)
```
