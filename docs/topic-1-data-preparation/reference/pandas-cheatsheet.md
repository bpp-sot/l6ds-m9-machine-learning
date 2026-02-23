# Pandas Data Prep Cheatsheet

> Quick reference for the most common Pandas operations used during Data Preparation.

## Reading Data

```python
import pandas as pd
import numpy as np
import seaborn as sns

# Load a built-in seaborn dataset effortlessly
df = sns.load_dataset('titanic')

# Read standard formats
csv_df = pd.read_csv('file.csv', na_values=['?'])
excel_df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
```

## Basic Inspection

```python
df.head(5)          # View first 5 rows
df.tail(3)          # View last 3 rows
df.shape            # Tuple of (rows, cols)
df.info()           # Column types, Non-Null counts, Memory usage
df.describe()       # Summary statistics (mean, std, min, max, quartiles)
df.columns          # List of column names
df.dtypes           # Data types of all columns
```

## Selecting & Filtering Columns

```python
# Select a single column (returns a Series)
ages = df['age']

# Select multiple columns (returns a DataFrame)
subset = df[['age', 'fare', 'survived']]

# Select columns algorithmically by mathematical datatype
numeric_df = df.select_dtypes(include=[np.number])
text_df = df.select_dtypes(include=['object', 'category'])
```

## Filtering Rows (Boolean Indexing)

```python
# Filter conditionally by value
adults = df[df['age'] >= 18]

# Filter using multiple exact conditions (use & for AND, | for OR)
female_survivors = df[(df['sex'] == 'female') & (df['survived'] == 1)]

# Filter using exactly matched lists
first_class = df[df['pclass'].isin([1, 2])]

# Filter natively using string matching
miss_titles = df[df['who'].str.contains('child')]
```

## Missing Data Operations

```python
df.isnull().sum()             # Count NaNs structurally per column
df.dropna()                   # EXTREMELY DANGEROUS: Drops any row with a NaN 
df.dropna(subset=['age'])     # Safely drop rows ONLY if 'age' is missing
df.dropna(thresh=10, axis=1)  # Drop specifically columns containing less than 10 non-NaN values

df['age'].fillna(30)          # Permanently replace ALL NaNs with 30
df['age'].fillna(df['age'].median()) # Replace with the median algorithmically
```

## Column Manipulation

```python
# Rename explicitly using dictionaries
df = df.rename(columns={'survived': 'Labels'})

# Drop permanently single columns
df = df.drop(columns=['Labels', 'age'])

# Typecast mathematically
df['fare'] = df['fare'].astype('float32')
```

## Grouping & Aggregating

```python
# Group securely by categorical text and calculate aggregates
grouped = df.groupby('sex').agg({
    'age': ['mean', 'max'],
    'fare': 'sum'
}).reset_index()
```

!!! tip "Workplace Tip"
    Bookmark this page during your EPA coding challenges. Quickly retrieving exact Pandas syntax for `.agg()` or `.select_dtypes()` under pressure demonstrates fluency and saves critical computational time during assessment interviews!
