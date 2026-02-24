# How to Load and Clean a Messy Dataset

> Real-world data is rarely formatted cleanly. Learn to quickly bypass bad metadata, skip rows, and coerce stubborn strings into workable floats.

## What You Will Learn
- Skip preamble metadata headers when reading datasets
- Handle inconsistent missing value indicators (`N/A`, `?`, `missing`)
- Convert currency strings (`$1,000`) into numeric columns instantly
- Rename columns programmatically into standard python snake_case

## Step 1: Handling Inconsistent Missing Indicators

When using built-in datasets like `sns.load_dataset('titanic')`, data is pre-cleaned. In real assessments, your stakeholders will upload files where missing values are marked inconsistently. You can handle this natively inside the `read_csv` parser.

```python
import pandas as pd
import numpy as np

# We simulate a messy CSV by creating a DataFrame with inconsistent missing values
messy_data = pd.DataFrame({
    'Passenger Id': [1, 2, 3],
    'Age': ['22', '?', 'N/A'],
    'Fare': ['$7.25', '$71.28', 'Missing']
})

# In the real world, you'd use pd.read_csv('messy.csv', na_values=['?', 'N/A', 'Missing'])
# Since we simulate the data, we use replace:
messy_data.replace(['?', 'N/A', 'Missing'], np.nan, inplace=True)

print(messy_data)
```

??? example "Expected Output"
    ```text
       Passenger Id   Age   Fare
    0             1    22  $7.25
    1             2   NaN  $71.28
    2             3   NaN    NaN
    ```

## Step 2: Fixing Column Names

Spaces and Capital Letters in column names break your ability to use dot-notation (e.g. `df.Age`). Convert them all cleanly using a lambda functional block to standardise them to `snake_case`.

```python
# Strip whitespace, lowercase, and replace spaces with underscores
messy_data.columns = [col.strip().lower().replace(' ', '_') for col in messy_data.columns]
print(messy_data.columns)
```

??? example "Expected Output"
    ```text
    Index(['passenger_id', 'age', 'fare'], dtype='object')
    ```

## Step 3: Coerce Strings to Numeric Types

Often currency or percentages load as object text (e.g. `$7.25`). We must strip the characters before converting them to floats.

```python
# Convert Age mathematically to numeric
messy_data['age'] = pd.to_numeric(messy_data['age'])

# Strip the '$' from Fare and convert computationally
messy_data['fare'] = messy_data['fare'].str.replace('$', '').astype(float)

print(messy_data.dtypes)
```

??? example "Expected Output"
    ```text
    passenger_id      int64
    age             float64
    fare            float64
    dtype: object
    ```

!!! tip "Workplace Tip"
    Do not iterate over rows manually using `for index, row in df.iterrows()` to fix strings! Python string vectorised operations like `.str.replace()` run exponentially faster natively in C underneath pandas and cleanly format 10 million rows in 0.5s.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5.3 | Common patterns in real-world data | Identifying missing values, duplicates, outliers, and class imbalance |
| S2 | Data engineering and governance | Systematic data cleaning, transformation, and quality assessment |
| S3 | Programming for data manipulation | pandas pipelines for data preparation |
| B3 | Adaptability and pragmatism | Handling imperfect real-world datasets |
