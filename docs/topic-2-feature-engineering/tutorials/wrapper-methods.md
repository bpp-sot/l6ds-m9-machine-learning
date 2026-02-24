# Wrapper Methods

> Wrapper methods actually train microscopic machine learning models recursively to test whether omitting a specific column damages predictive velocity.

## What You Will Learn
- Differentiate Wrapper methods from Filter methods computationally
- Deploy Recursive Feature Elimination (RFE)
- Assess the radical computation costs involved

## Prerequisites
- Completed the *Filter Methods* tutorial
- Basic understanding of standard supervised classifiers (`LogisticRegression`)

## Step 1: The Flaw in Filtering

Filter methods evaluate columns exclusively individually. If `Age` is highly predictive, and `Date of Birth` is highly predictive, a Filter method will stubbornly keep *both*. But keeping both is redundant multicollinearity.

Wrapper Methods solve this. They actually train a model with `Age`, then train a totally new model with `Date of Birth`, explicitly noticing that removing one doesn't alter accuracy.

## Step 2: Recursive Feature Elimination (RFE)

RFE builds an algorithm utilizing every single feature. It ranks all features by validation weight coefficients. It executes the weakest feature, drops the column, and mathematically retrains the entire model dynamically from scratch cleanly until `n_features_to_select` remains.

We will systematically eliminate components of the `diamonds` array utilizing a Logistic model.

```python
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample the dataset for extreme speed 
df = sns.load_dataset('diamonds').sample(500, random_state=42)

# Prepare pure numeric Data
X = df[['carat', 'depth', 'table', 'price']]
y = LabelEncoder().fit_transform(df['cut']) # Encode text target for classification

# Linear algorithms require standardisation natively!
X_scaled = StandardScaler().fit_transform(X)

# We initialize the "Base Estimator" (the model that iterates)
eval_model = LogisticRegression(max_iter=1000)

# We initialize RFE, telling it to strip down to precisely 2 features
selector = RFE(estimator=eval_model, n_features_to_select=2, step=1)

# Execute the recursive loop computationally
selector.fit(X_scaled, y)

# Inspect the outputs
results = pd.DataFrame({
    'Feature': X.columns,
    'Elimination Rank': selector.ranking_,
    'Selected': selector.support_
}).sort_values(by='Elimination Rank')

print(results)
```

??? example "Expected Output"
    ```text
      Feature  Elimination Rank  Selected
    2   table                 1      True
    0   carat                 1      True
    1   depth                 2     False
    3   price                 3     False
    ```

In this simulation, the algorithm structurally dropped `price` immediately (Rank 3), subsequently dropped `depth` (Rank 2) explicitly preserving only `table` and `carat` as the maximal synergistic combination dimensions universally!

!!! tip "Workplace Tip"
    Do not use RFE directly against datasets possessing 5,000 columns. RFE structurally retrains an entire Machine Learning model exactly 5,000 times! Utilize extremely fast **Filter Methods** dynamically to slice your dimensions from 5,000 to 100, and *then* run RFE carefully on the surviving 100 constraints.

## Summary
- **Wrapper Methods** utilize operational Machine Learning algorithms natively to execute scoring.
- They intelligently discover explicit collinearity flaws that singular arbitrary Filter Methods universally ignore natively.
- **Recursive Feature Elimination (RFE)** loops persistently backwards, terminating the singularly weakest geometric parameter mathematically each pass.

## Next Steps
→ [Embedded Methods](embedded-methods.md) — algorithmically combining the calculation speed natively of Filters efficiently with the holistic precision of Wrappers concurrently!

??? challenge "Stretch & Challenge"
    ### For Advanced Learners
    
    **Forward Sequential Selection (SFS)**
    
    RFE calculates backwards recursively (starting with 100, dropping down explicitly to 10). Forward Sequential Selection works identically but starts completely empty!
    
    It trains 100 models possessing exactly 1 column each. It locates the singular best parameter, freezes it dynamically into the list, and tests 99 models containing exactly 2 arrays. Look closely into `from sklearn.feature_selection import SequentialFeatureSelector`.
    
    It is catastrophically unscalable, but explicitly guarantees locating the singular cleanest minimal operational parameter subset cleanly!

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced analytics and ML techniques | Feature selection algorithms and dimensionality reduction |
| K5.2 | Data formats and structures | Encoding categorical variables, handling mixed feature types |
| S2 | Data engineering | Creating and transforming features from raw data |
| S4 | Feature selection and ML | Applying feature selection methods and PCA |
| B1 | Inquisitive approach | Exploring creative feature engineering strategies |
