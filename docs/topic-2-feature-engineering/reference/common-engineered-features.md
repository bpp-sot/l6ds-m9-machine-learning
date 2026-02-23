# Common Engineered Features Cheat Sheet

> Need inspiration for feature engineering? Here are the most common high-signal mathematical transformations utilized universally by Data Scientists.

## Temporal Features (Datetime)

Derived purely mechanically from strictly one Timestamp column.

| Feature Type | How to Create | Why it Matters |
|---|---|---|
| **Cyclical Units** | `df['date'].dt.month` | Identifies human seasonal behavior geometrically |
| **Duration/Elapsed** | `(today - df['date']).dt.days` | Absolute chronologies (e.g., "Days Since Last Login") |
| **Time of Day** | `df['date'].dt.hour` | High predictive signal logically for user engagement bursts |
| **Is Weekend** | `df['date'].dt.dayofweek >= 5` | Binary switch tracking structural behavior breaks |

## Mathematical Ratios

Constructed structurally by mechanically dividing two continuous variables algorithmically.

| Feature Type | How to Create | Why it Matters |
|---|---|---|
| **Per Capita** | `df['GDP'] / df['population']` | Scales massive absolute volume into comparative individual density |
| **Percentage Change** | `df['q4_sales'] / df['q3_sales']` | Identifies acceleration vectors physically independently of scale |
| **Proportions** | `df['bedrooms'] / df['total_rooms']` | Uncovers geometric property layouts irrespective of house dimension |

## Domain Boundaries (Binning)

Compressing structurally dispersed variance algebraically into logical categorical intervals explicitly.

| Feature Type | How to Create | Why it Matters |
|---|---|---|
| **Generational Cohorts** | `pd.cut(df['age'], bins=[0, 18, 35, 65])` | Forces predictive algorithms mechanically to respect sociological reality natively |
| **Pricing Tiers** | `pd.qcut(df['price'], q=4)` | Divides natively continuous sales equally mechanically into "Budget, Mid, Premium, Luxury" quartiles |

## String Metadata (Natural Language)

Bypassing extreme computationally intense NLP vectorization by aggressively tracking structural text dimensions recursively.

| Feature Type | How to Create | Why it Matters |
|---|---|---|
| **Text Length** | `df['text'].str.len()` | Longer reviews geometrically explicitly correlate natively with anger or passion structurally |
| **Word Count** | `df['text'].apply(lambda x: len(str(x).split()))` | Measures density over length structurally |
| **Title Presence** | `df['name'].str.contains('Dr.')` | Extracts binary categorical demographic data deeply embedded natively within raw input arrays |

!!! tip "Workplace Tip"
    Whenever a Dataset physically lacks robust initial continuous independent variables structurally, deriving "Elapsed Duration" natively and "Mathematical Proportions" geometrically generates exactly the high-signal predictive tensors that Machine Learning classifiers implicitly crave mechanically!
