# How-to: Handle High-Cardinality Categories

## The Problem
If a feature like "City" or "Postcode" has hundreds or thousands of unique string values, utilizing One-Hot Encoding (`pd.get_dummies`) creates an impossibly wide, sparse dataset resulting in the "Curse of Dimensionality" and memory crashes.

## The Solution
You must restrict or transform categorical values.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder

# Generate high-cardinality data
categories = ['Cat_A']*50 + ['Cat_B']*30 + ['Cat_C']*15 + ['Small_Cat']*5
np.random.shuffle(categories)
df = pd.DataFrame({
    'Feature': categories,
    'Target': np.random.randint(0, 2, 100) # Binary classification target
})

# Approach 1: Frequency Grouping (The "Other" Bucket)
# Look at frequencies
counts = df['Feature'].value_counts()
print(f"Original Categories:\\n{counts}")

# Decide a threshold (e.g. keep top 2, group the rest)
top_categories = counts.nlargest(2).index
df['Reduced_Feature'] = df['Feature'].where(df['Feature'].isin(top_categories), 'Other')

print(f"\\nReduced Categories:\\n{df['Reduced_Feature'].value_counts()}")

# Approach 2: Target Encoding
# Replaces string category with the Mean Target Value for that category
# Excellent for Postcodes, Store IDs, Vehicle Models
encoder = TargetEncoder(smooth="auto")

# Must reshape to 2D array for sklearn
df['Target_Encoded'] = encoder.fit_transform(df[['Feature']], df['Target'])
print(f"\\nEncoded Mapping Preview:\\n{df.groupby('Feature')['Target_Encoded'].mean()}")
```

## Discussion

### Feature Hashing
An alternative for Neural Networks or Massive scales (streaming text categorization) is `FeatureHasher` from `sklearn.feature_extraction`. It hashes strings to bounded buckets, preventing the feature space from expanding infinitely, though it loses interpretability because you can no longer trace "Hash Column 3" back to "Manchester".

### Caveats
- Using `TargetEncoder` introduces extreme risks for **Data Leakage**. You must strictly fit the encoder *only* on the training data. If you implement this in your L6 Assessment, explicitly document your leakage mitigation strategy using `Pipeline()`.
