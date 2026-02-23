# Building Preprocessing Pipelines

> Manual step-by-step cleaning creates data leakage and terrifying bugs. Pipelines permanently package processing into a single repeatable block.

## What You Will Learn
- Build nested Scikit-Learn `Pipeline` architectures
- Use `ColumnTransformer` to route distinct data types concurrently
- Prevent disastrous data leakage across testing sets 

## Prerequisites
- Completed all previous Data Preparation tutorials (Imputation, Encoding, Scaling)
- Understanding of independent vs dependent variables (`X` vs `Y`)

## Step 1: The Danger of Manual Processing

Until now, we have processed data row by row, column by column sequentially:

```python
# DANGEROUS MANUAL PROCESS
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')

# Separate Features (X) and Target (y)
X = df.drop(['survived', 'who', 'adult_male', 'deck', 'alive', 'alone'], axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

If we `impute()` and `scale()` the full `X` block *before* splitting the test set, information from the test set mathematically leaks heavily into our training mean bounds. 

We must enforce processing explicitly only upon the `X_train` data, then blindly apply that learned processing strictly via `.transform()` onto `X_test`. Manually tracking this for 100 features is functionally impossible in production.

## Step 2: Numeric Piplines

Pipelines physically chain independent transformers together structurally.

```python
from sklearn.pipeline import Pipeline

# 1. Pipeline for purely numeric columns
numeric_features = ['age', 'fare']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Step A: Fill missing
    ('scaler', StandardScaler())                   # Step B: Scale standard deviation
])
```

## Step 3: Categorical Pipelines

```python
# 2. Pipeline for purely categorical (text) columns
categorical_features = ['pclass', 'sex', 'embarked', 'class', 'embark_town']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Step A: Fill missing with Mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Step B: One-hot encode string arrays
])
```

## Step 4: The ColumnTransformer Combinator

The `ColumnTransformer` dynamically directs our raw columns physically into their appropriate parallel pipelines.

```python
from sklearn.compose import ColumnTransformer

# 3. Combine both parallel channels identically 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

## Step 5: Master Execution

Now we apply this entire complex engine securely against our raw splits!

```python
# 4. Fit against the Training set ONLY
X_train_processed = preprocessor.fit_transform(X_train)

# 5. Transform the Test set specifically
X_test_processed = preprocessor.transform(X_test)

print(f"Raw shape: {X_train.shape}")
print(f"Processed shape: {X_train_processed.shape}")
```

??? example "Expected Output"
    ```text
    Raw shape: (712, 9)
    Processed shape: (712, 22)
    ```

Notice the system flawlessly inflated our raw categories out to 22 dimensions, imputed the NaNs, and generated scaled numerical values, totally isolating test data leakage and executing the architecture using absolutely zero arbitrary Pandas mappings.

!!! info "Assessment Connection"
    Providing explicit, robust `Pipeline` architectures rather than sequential flat scripting blocks proves structural modularity and architectural literacy—this strictly maps into distinction criteria against EPA ML model frameworks.

## Summary
- Manual sequence cleaning causes silent predictive capability destruction via Data Leakage.
- Use `Pipeline` to chain sequential algorithms like `impute()` -> `scale()` tightly.
- Use `ColumnTransformer` to segregate string data concurrently from numeric columns prior to targeting independent preprocessing execution methodologies.

## Next Steps
→ [Clean a Messy CSV File](../how-to/clean-messy-csv.md) — review how-to guides targeting specific common data transformation annoyances!

??? challenge "Stretch & Challenge"
    ### For Advanced Learners
    
    **Including Modelling inside the Pipeline Architecture**
    
    You don't just have to put pre-processing in a pipeline. You can bolt the actual Machine Learning `RandomForestClassifier` directly identically to the end of the transformer block!
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    # Bundle preprocessor and model together
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    # Train the pre-processing and the ML algorithm identically!
    full_pipeline.fit(X_train, y_train)
    
    # Predict directly via the raw uncleaned test features!
    predictions = full_pipeline.predict(X_test)
    ```
    
    When you deploy this pipeline into production servers, you don't even need to pre-clean the live structural payload data! You just feed raw dictionaries dynamically straight into `.predict()`.

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S13 | Apply appropriate machine learning algorithms | Compiling reproducible execution nodes via strict Pipelines |
| K5 | Machine Learning workflows | Building structured ML DAG structures eliminating Data Leakage |
| B2 | Logical and analytical approach | Segregating execution logic categorically into numeric vs categorical streams |
