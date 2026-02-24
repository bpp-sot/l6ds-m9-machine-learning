# Datetime & Text Features

> Dates and sentences are unreadable by ML algorithms. We must shatter them into numerical matrices.

## What You Will Learn
- Extract Cyclical data (Months, Weeks, Hours) from raw Datetime strings
- Compute Elapsed Time features natively in Pandas
- Derive word counts and string lengths computationally 

## Prerequisites
- Completed the *Creating Features* tutorial
- Basic understanding of Python `datetime` objects

## Step 1: Shattering Timestamps

A raw timestamp like `2024-12-25 10:30:00` is read by Pandas as an `object` string. To a Machine Learning algorithm, this is indistinguishable from the text `Apple`. We must convert the string into a chronologically aware mathematical tensor using `pd.to_datetime()`.

We will utilize the built-in `flights` Seaborn dataset to demonstrate unpacking granular time arrays.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('flights')

# Synthesise a proper timestamp from the year and month
df['timestamp'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
print(f"Data type: {df['timestamp'].dtype}")
```

??? example "Expected Output"
    ```text
    Data type: datetime64[ns]
    ```

Once cast to `datetime64[ns]`, we can tap the absolute power of the Pandas `.dt` accessor to extract dozens of numerical features instantaneously.

```python
# Engineer numeric cyclical columns explicitly
df['day_of_week'] = df['timestamp'].dt.dayofweek # 0=Monday, 6=Sunday
df['quarter'] = df['timestamp'].dt.quarter
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

print(df[['timestamp', 'day_of_week', 'quarter', 'is_weekend']].head())
```

??? example "Expected Output"
    ```text
       timestamp  day_of_week  quarter  is_weekend
    0 1949-01-01            5        1           1
    1 1949-02-01            1        1           0
    2 1949-03-01            1        1           0
    3 1949-04-01            4        2           0
    4 1949-05-01            6        2           1
    ```

Your model can now algorithmically discover that passenger volume explodes when `quarter = 2` without needing any conception of what "Summer" physically is!

## Step 2: Elapsed Time

Often, the explicit Date isn't the predictive signal; the *duration* from a specific event is. For customer churn, predicting off "Registration Date" is useless. Predicting off "Days Since Last Login" is critical.

```python
# Assuming today is Jan 1st, 1961
current_date = pd.to_datetime('1961-01-01')

# Calculate mathematical timedelta (ElapsedTime)
df['days_elapsed'] = (current_date - df['timestamp']).dt.days

print(df[['timestamp', 'days_elapsed']].tail())
```

??? example "Expected Output"
    ```text
         timestamp  days_elapsed
    139 1960-08-01           153
    140 1960-09-01           122
    141 1960-10-01            92
    142 1960-11-01            61
    143 1960-12-01            31
    ```

## Step 3: Text Length Engineering

When NLP (Natural Language Processing) is too computationally expensive, you can extract structural metadata from raw strings. For spam classification, the length of the email text is often a stronger signal than the actual words!

```python
# Synthetic sample of customer reviews
reviews = pd.DataFrame({
    'text': [
        "Great product!", 
        "Terrible experience, shipping was late, box arrived completely shattered, returning immediately.",
        "Okay."
    ]
})

# Engineer string metadata using the .str accessor
reviews['char_count'] = reviews['text'].str.len()
reviews['word_count'] = reviews['text'].str.split().str.len()

print(reviews)
```

??? example "Expected Output"
    ```text
                                                    text  char_count  word_count
    0                                     Great product!          14           2
    1  Terrible experience, shipping was late, box ar...          94          12
    2                                              Okay.           5           1
    ```

A model can now instantly separate extreme anger (Review 1) from general satisfaction (Review 0) purely based on `word_count`, knowing nothing about the English language.

!!! tip "Workplace Tip"
    When utilizing `day_of_week` (0-6) or `month` (1-12) as features, Linear regressions treat them maliciously! A linear model maps December (12) as being *furthest away* from January (1). In reality, they are adjacent. Cyclical encoding utilizing `sin` and `cos` algorithms is the formal industry standard for time loops.

## Summary
- Convert strings rapidly securely into `datetime64` arrays to utilize the native `.dt` parser.
- Shattering one Timestamp column generates explicitly 5+ high-signal numeric arrays (Day, Month, Is_Weekend).
- Calculate absolute duration metrics continuously from fixed chronological anchors.
- Bypass heavy NLP models safely by relying on simple `.str.len()` metadata signals for string matrices.

## Next Steps
→ [Filter Methods](filter-methods.md) — having created 50 new features, we must structurally select only the most heavily predictive.

??? challenge "Stretch & Challenge"
    ### For Advanced Learners
    
    **Cyclical Sine/Cosine Encoding**
    
    To solve the Linear Regression "December vs January" bug mentioned in the Workplace Tip, we mathematically project the 12 months onto a continuous circular trigonometric function.
    
    ```python
    import numpy as np
    
    # We must explicitly map the feature geometrically out to a 2D circle
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    ```
    
    Now, Dec (Month 12) and Jan (Month 1) share highly identical geometric spatial coordinates natively on the mapped algorithm axis! Research explicitly why this works for Neural Networks dealing with 24-hour retail sales metrics.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Feature selection algorithms and dimensionality reduction |
| K5.2 | Data formats and structures | Encoding categorical variables, handling mixed feature types |
| S2 | Data engineering | Creating and transforming features from raw data |
| S4 | Feature selection and ML | Applying feature selection methods and PCA |
| B1 | Inquisitive approach | Exploring creative feature engineering strategies |
