# How to Handle High Cardinality Features

> When a text column contains hundreds of unique classes (e.g. zip codes, product names), traditional encoding explodes into a Curse of Dimensionality.

## What You Will Learn
- Group highly fragmented classes into "Other"
- Execute Frequency Encoding
- Execute Target Encoding 

## Step 1: Grouping Rare Tails ("Othering")

If a column like `City` has `London` occurring 5000 times, and 400 rural villages occurring once each, mathematically mapping those 400 villages provides zero predictive lift and massively confuses algorithms. Group them together into a generic `Other` bucket.

```python
import pandas as pd
import seaborn as sns

# We construct synthetic data with a 'long tail' of rare occurrences
data = {'City': ['London']*1000 + ['Manchester']*500 + ['Leeds']*100 + ['VillageA', 'VillageB', 'VillageC', 'TownD', 'TownE']}
df = pd.DataFrame(data)

# Find the frequency of each 
freqs = df['City'].value_counts()

# Define an arbitrary threshold (e.g. must appear > 50 times)
rare_cities = freqs[freqs <= 50].index

# Replace them all with 'Other'
df.loc[df['City'].isin(rare_cities), 'City'] = 'Other'

print(df['City'].value_counts())
```

??? example "Expected Output"
    ```text
    City
    London        1000
    Manchester     500
    Leeds          100
    Other            5
    ```

## Step 2: Frequency Encoding

Instead of one-hot encoding 50 classes into 50 sparse columns, you can simply replace the text with the exact integer frequency of how often it appears. 

```python
# The frequency map
freq_map = df['City'].value_counts().to_dict()

# Map the raw integers back into the column
df['City_Freq_Encoded'] = df['City'].map(freq_map)

print(df.drop_duplicates())
```

??? example "Expected Output"
    ```text
                City  City_Freq_Encoded
    0         London               1000
    1000  Manchester                500
    1500       Leeds                100
    1600       Other                  5
    ```

Decision Trees (`RandomForest`, `XGBoost`) process Frequency Encoded columns efficiently directly, naturally interpreting that `London (1000)` structurally behaves identically to standard numeric parameters.

## Step 3: Target Encoding

Target encoding permanently maps the categorical string to the *historical mean* of the dependent target variable.

```python
# Assuming we are trying to predict Salary
df['Salary'] = [50000]*1000 + [40000]*500 + [35000]*100 + [30000]*5 

# Using category_encoders
import category_encoders as ce

encoder = ce.TargetEncoder(cols=['City'])
df['City_Target_Encoded'] = encoder.fit_transform(df['City'], df['Salary'])

print(df[['City', 'City_Target_Encoded']].drop_duplicates())
```

??? example "Expected Output"
    ```text
                City  City_Target_Encoded
    0         London         50000.000000
    1000  Manchester         40000.000000
    1500       Leeds         35000.000000
    1600       Other         30588.235294
    ```

!!! tip "Workplace Tip"
    Target Encoding is strictly dependent on the training `Y` label. Be exceptionally careful not to accidentally `.fit_transform()` your live Test sets or Production records using this encoder. It creates the most catastrophic Data Leakage bugs in the industry!

## KSB Mapping

| KSB | Description | How This Guide Addresses It |
|-----|-------------|-------------------------------|
| S12 | Feature engineering | Mitigating Curse of Dimensionality utilizing dense algebraic encoding methodologies |
| K6 | Data analytics and visualisation | Profiling class imbalances and distributions statistically |
