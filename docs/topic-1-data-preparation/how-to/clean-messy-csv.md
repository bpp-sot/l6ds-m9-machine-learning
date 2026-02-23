# How-to: Clean a Messy CSV File

## The Problem
In your workplace projects, data extraction rarely returns a pristine CSV. Frequently, your data will contain unstructured columns, mixed delimiters, erratic spacing, or hidden artifacts.

## The Solution

Here are the most robust Pandas commands to quickly sanitize raw structure.

```python
import pandas as pd
import numpy as np

# 1. Handling Bad Delimiters or Encodings
# If your file errors on loading due to commas in strings or strange characters:
df_raw = pd.read_csv('messy_file.csv', 
                     delimiter=';',          # European format
                     encoding='cp1252',      # Common Windows DB export
                     skiprows=2)             # Skip metadata headers

# 2. Stripping Whitespace
# " London " and "London" are entirely different strings to a machine!
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')

# Strip whitespace from ALL string columns using applymap
df_clean = df_raw.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 3. Fixing Corrupted Currencies/Numbers
# For example: "$1,500.00" must be converted to float 1500.00
if 'revenue' in df_clean.columns:
    df_clean['revenue'] = df_clean['revenue'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# 4. Standardizing Missing Formats
# Missing data might look like "N/A", "Unknown", "?", or "-99"
df_clean.replace(['N/A', 'Unknown', '?', -99], np.nan, inplace=True)

print(df_clean.head())
```

## Discussion

### When to use this approach?
Apply these cleanup sequences *before* utilizing `SimpleImputer` or `StandardScaler`. MLOps pipelines expect structurally sound tables.

### Caveats
- Regex replacements (`replace({'\$': ''}, regex=True)`) are computationally expensive on datasets exceeding 10M rows. If possible, process massive strings using Dask or PySpark.
- Replacing "-99" with `np.nan` assumes "-99" genuinely means missing. If "-99" is a valid code (e.g., negative balance), you will destroy your model. Always inspect distributions before automating!
