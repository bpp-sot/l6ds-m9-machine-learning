# Loading & Exploring Data

> The first step in any ML project is understanding what you're working with.

## What You Will Learn

- Load data from CSV, Excel, and SQL sources using pandas
- Inspect dataset shape, types, and basic statistics
- Identify missing values and duplicates
- Create initial visualisations to understand distributions and relationships
- Document your Exploratory Data Analysis (EDA) findings

## Prerequisites

- Python environment set up ([see setup guide](../../getting-started/setup.md))
- Basic familiarity with pandas and matplotlib

## Step 1: Load Your Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
sns.set_style('whitegrid')
```

### From CSV

```python
df = pd.read_csv('data.csv')
```

### From Excel

```python
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

### From SQL

```python
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM customers', conn)
```

!!! tip "Workplace Tip"
    In your workplace project, document exactly where your data comes from. The assessment rubric values transparency about data sources and any transformations applied before analysis.

## Step 2: First Look at the Data

```python
# Shape: rows x columns
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# First few rows
df.head()
```

```python
# Column names and data types
df.info()
```

```python
# Summary statistics for numeric columns
df.describe()
```

```python
# Summary statistics for categorical columns
df.describe(include='object')
```

!!! note "Assessment Connection"
    Section A of your presentation should demonstrate that you thoroughly understood your data before modelling. Examiners want to see evidence of systematic exploration, not just jumping straight to algorithms.

## Step 3: Missing Values Audit

```python
# Count missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
}).sort_values('Missing %', ascending=False)

print(missing_summary[missing_summary['Missing Count'] > 0])
```

### Visualise Missing Patterns

```python
import missingno as msno

# Matrix view — shows patterns of missingness
msno.matrix(df, figsize=(12, 6))
plt.title('Missing Value Patterns')
plt.tight_layout()
plt.show()
```

```python
# Heatmap — shows correlations between missing values
msno.heatmap(df, figsize=(10, 6))
plt.title('Missing Value Correlations')
plt.tight_layout()
plt.show()
```

!!! tip "Workplace Tip"
    Missing data is rarely random. If two columns are missing together, there's often a business reason. For example, in HR data, 'promotion_date' and 'new_salary' might both be missing for employees who weren't promoted.

## Step 4: Duplicates Check

```python
# Check for exact duplicate rows
n_duplicates = df.duplicated().sum()
print(f"Exact duplicate rows: {n_duplicates}")

if n_duplicates > 0:
    print("\nFirst few duplicates:")
    print(df[df.duplicated(keep=False)].head(10))
```

```python
# Check for duplicates on key columns
key_cols = ['customer_id', 'date']  # Adjust for your dataset
n_key_dupes = df.duplicated(subset=key_cols).sum()
print(f"Duplicates on key columns: {n_key_dupes}")
```

## Step 5: Distribution Analysis

### Numeric Variables

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(
    nrows=len(numeric_cols),
    ncols=2,
    figsize=(14, 4 * len(numeric_cols))
)

for i, col in enumerate(numeric_cols):
    # Histogram
    axes[i, 0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[i, 0].set_title(f'{col} — Distribution')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')

    # Box plot
    axes[i, 1].boxplot(df[col].dropna(), vert=True)
    axes[i, 1].set_title(f'{col} — Box Plot')

plt.tight_layout()
plt.show()
```

### Categorical Variables

```python
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Top 5 values:")
    print(df[col].value_counts().head())
```

## Step 6: Correlation Analysis

```python
# Correlation matrix
corr_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5
)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

```python
# Pairplot for key features (limit to 5–6 columns for readability)
key_features = ['feature1', 'feature2', 'feature3', 'target']  # Adjust
sns.pairplot(df[key_features], diag_kind='kde', corner=True)
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()
```

## Step 7: Document Your Findings

Create an EDA summary to include in your presentation:

```python
eda_summary = {
    'Dataset': 'Your dataset name',
    'Rows': df.shape[0],
    'Columns': df.shape[1],
    'Numeric features': len(df.select_dtypes(include=[np.number]).columns),
    'Categorical features': len(df.select_dtypes(include='object').columns),
    'Missing values': df.isnull().sum().sum(),
    'Duplicate rows': df.duplicated().sum(),
    'Target variable': 'target_column_name',
    'Key observations': [
        'Observation 1 about your data',
        'Observation 2 about distributions',
        'Observation 3 about relationships',
    ]
}

for k, v in eda_summary.items():
    print(f"{k}: {v}")
```

!!! success "What Good EDA Looks Like in Your Assessment"
    The difference between a pass and a distinction often lies in the quality of your EDA. Document:

    - **What you found** — key patterns, anomalies, relationships
    - **What it means** — business interpretation of the patterns
    - **What you'll do about it** — how findings inform your preprocessing and modelling decisions

## Summary

In this tutorial you learned to:

- Load data from multiple sources using pandas
- Systematically inspect shape, types, and statistics
- Audit missing values with counts, percentages, and visual patterns
- Check for and handle duplicate records
- Visualise distributions and identify potential outliers
- Analyse correlations between features
- Document EDA findings for your assessment

## Next Steps

→ [Handling Missing Values](missing-values.md) — decide how to treat the missing data you've identified

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| K3 | Data management and storage | Loading data from multiple sources |
| K6 | Data analytics and visualisation | EDA visualisations and statistical summaries |
| S4 | Import, cleanse, transform data | Systematic data quality assessment |
| S7 | Analyse data to generate insights | Interpreting distributions and correlations |
| B2 | Logical and analytical approach | Structured, documented exploration process |
