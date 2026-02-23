# DateTime & Text Features

> "Time is a highly structured construct. A machine learning model seeing '2023-11-23' merely sees a string of characters unless you teach it the concept of 'Thursday'."

## What You Will Learn

- Extract cyclical components from temporal data (Month, Day of Week, Is_Weekend)
- Handle Textual transformations using `TF-IDF` (Term Frequency-Inverse Document Frequency)
- Identify unstructured sentiment

## Step 1: Unpacking DateTime Objects

Algorithms cannot mathematically process `2023-11-23 14:00:00`. We must deconstruct this into numeric parts so the AI can capture temporal patterns (e.g., "Sales always spike on weekends" or "Traffics dips in August").

```python
import pandas as pd
import numpy as np

# Sample transactional timestamps
df = pd.DataFrame({
    'Transaction_Time': ['2023-12-25 08:30:00', '2024-01-15 14:45:00', '2024-03-03 21:00:00']
})

# Convert string sequence to native Pandas datetime format
df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'])

# Feature Extraction
df['Year'] = df['Transaction_Time'].dt.year
df['Month'] = df['Transaction_Time'].dt.month
df['DayOfWeek'] = df['Transaction_Time'].dt.dayofweek   # 0 = Monday, 6 = Sunday
df['Hour'] = df['Transaction_Time'].dt.hour
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

print(df.drop('Transaction_Time', axis=1))
```

!!! note "Assessment Connection"
    In time-series models or regression tasks on your apprenticeship exam, temporal extraction is foundational. Extracting `Is_Weekend` or `Quarter` often yields the highest Model Feature Importance in final business evaluations.

## Step 2: Cyclical Encoding for Time

A glaring flaw with the standard extraction above: The algorithm thinks `Month=12` (December) is as far away as possible from `Month=1` (January). In reality, they are adjacent.

We solve this using **Sine and Cosine Transforms** to map the date onto a circle.

\\[
X_{sin} = \\sin\\left(\\frac{2 \\pi X}{max(X)}\\right) \\quad X_{cos} = \\cos\\left(\\frac{2 \\pi X}{max(X)}\\right)
\\]

```python
# Assuming max month is 12
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Now, Month 1 and Month 12 are geographically close in the newly mapped feature space.
```

## Step 3: Structuring Text with TF-IDF

If your dataframe contains raw strings of text (e.g., Customer Reviews), you can use `TfidfVectorizer` to quantify the vocabulary.

**TF-IDF** stands for Term Frequency - Inverse Document Frequency. It mathematically scores words based on frequency. Words that appear constantly everywhere (like "the", "and") receive scores near 0, while unique, identifying words ("terrible", "outstanding") receive high scores.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    "The product arrived broken.",
    "Outstanding quality and fast shipping.",
    "The product works okay but nothing outstanding."
]

# Stop_words='english' removes "the", "and", "is", etc.
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# Convert to a readable dataframe
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df)
```

## Summary

Handling specialized types (Dates and Text) transforms messy human inputs into high-value engineered predictors. 

## Next Steps

→ [Filter Selection Methods](filter-methods.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S7 | Analyse logical insights | Generating categorical indicators out of unstructured Text |
| K3 | Data management logic | Operating `datetime` and vectorized structures |
