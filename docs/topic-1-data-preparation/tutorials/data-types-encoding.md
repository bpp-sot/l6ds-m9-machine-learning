# Data Types & Encoding

> Algorithms only understand numbers. Encoding is how you translate categorical text into a mathematical format a model can learn from.

## What You Will Learn
- Differentiate between Nomimal and Ordinal data types
- Apply Ordinal Encoding to hierarchical categories quickly
- Apply One-Hot Encoding to unranked categorical data
- Protect your pipeline against the "Dummy Variable Trap"

## Prerequisites
- Basic understanding of Pandas DataFrames
- Completed the *Handling Missing Values* tutorial

## Step 1: Loading Categorical Data

We will use the `diamonds` dataset from Seaborn, which is famous for its rich mix of categorical features detailing the physical cut, colour, and clarity of diamonds.

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset('diamonds').head(1000) # Sampled for speed
print(f"Categorical features: \n{df.select_dtypes(include='category').head()}")
```

??? example "Expected Output"
    ```text
    Categorical features: 
         cut color clarity
    0  Ideal     E     SI2
    1  Premium   E     SI1
    2  Good      E     VS1
    3  Premium   I     VS2
    4  Good      J     SI2
    ```

## Step 2: Ordinal Encoding (Hierarchical Data)

If the text data has an inherent ranking or order (e.g. `Good < Premium < Ideal`), you must use Ordinal Encoding to preserve that hierarchy as sequential integers.

```python
from sklearn.preprocessing import OrdinalEncoder

# The 'cut' column has a strict ranking
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

# We must explicitly pass the hierarchy to the encoder as a 2D list
encoder = OrdinalEncoder(categories=[cut_categories])

# Transform the column
df['cut_encoded'] = encoder.fit_transform(df[['cut']])

print(df[['cut', 'cut_encoded']].drop_duplicates().sort_values('cut_encoded'))
```

??? example "Expected Output"
    ```text
             cut  cut_encoded
    18      Fair          0.0
    2       Good          1.0
    5  Very Good          2.0
    1    Premium          3.0
    0      Ideal          4.0
    ```

!!! tip "Workplace Tip"
    Never use implicit `LabelEncoder` for features. Explicitly defining the hierarchy ensures that a missing category in your training set doesn't shift the integer scores of the remaining categories during production inference.

## Step 3: One-Hot Encoding (Nominal Data)

If categories have no inherent ranking (e.g. Colour `E` vs `J`), mapping them to `1` and `2` implies that `J` mathematically is "double" `E`, which confuses linear algorithms. Instead, we use One-Hot Encoding to split the category into distinct True/False binary columns.

```python
from sklearn.preprocessing import OneHotEncoder

# 'color' has no strict hierarchy, it is purely nominal
ohe = OneHotEncoder(sparse_output=False, drop='first') # Drop first prevents collinearity

# Fit and transform
color_encoded = ohe.fit_transform(df[['color']])

# Get the generated column names
new_columns = ohe.get_feature_names_out(['color'])

# Convert back to a DataFrame for viewing
df_ohe = pd.DataFrame(color_encoded, columns=new_columns)
print(df_ohe.head())
```

??? example "Expected Output"
    ```text
       color_E  color_F  color_G  color_H  color_I  color_J
    0      1.0      0.0      0.0      0.0      0.0      0.0
    1      1.0      0.0      0.0      0.0      0.0      0.0
    2      1.0      0.0      0.0      0.0      0.0      0.0
    3      0.0      0.0      0.0      0.0      1.0      0.0
    4      0.0      0.0      0.0      0.0      0.0      1.0
    ```

!!! info "Assessment Connection"
    Setting `drop='first'` prevents the **Dummy Variable Trap**. If a diamond is not `E`, `F`, `G`, `H`, `I`, or `J`, the algorithm automatically deduces the remaining colour is `D` (all zeroes). Keeping `D` as a column creates perfect multicollinearity, which destroys Linear Regression weights. Explain this trap concisely in your EPA to secure higher marks.

## Summary
- Use **OrdinalEncoder** precisely when a hierarchy is explicitly defined in business logic (`Small < Medium < Large`).
- Pass the explicit sequential array to your `OrdinalEncoder` rather than letting it guess alphabetically.
- Use **OneHotEncoder** for unranked strings (cities, species, IDs).
- Always `drop='first'` in One-Hot Encoding to prevent matrix collinearity in linear models.

## Next Steps
→ [Scaling & Normalisation](scaling-normalisation.md) — harmonise the mathematical weight of distinct continuous distributions.

??? challenge "Stretch & Challenge"
    ### For Advanced Learners
    
    **Target Encoding for High Cardinality**
    
    If you have a column with a massive number of unique categorical text features (e.g. `Postcode` or `Make/Model`), One-Hot Encoding will generate hundreds of empty sparse columns (the Curse of Dimensionality).
    
    Instead, algorithms can replace the text with the *mean target value* of that category. 
    
    ```python
    import category_encoders as ce
    
    # Target encoding replaces the text feature with the 
    # historical mean price of diamonds with that specific colour
    target_encoder = ce.TargetEncoder(cols=['color'])
    df['color_target_encoded'] = target_encoder.fit_transform(df['color'], df['price'])
    ```
    
    This concisely collapses 50+ categorical classes into a single dense numeric feature with high predictive signal. Have a look into `category_encoders.TargetEncoder` to elevate your predictive capability!

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5.3 | Common patterns in real-world data | Identifying missing values, duplicates, outliers, and class imbalance |
| S2 | Data engineering and governance | Systematic data cleaning, transformation, and quality assessment |
| S3 | Programming for data manipulation | pandas pipelines for data preparation |
| B3 | Adaptability and pragmatism | Handling imperfect real-world datasets |
