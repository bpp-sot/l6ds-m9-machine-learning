# How to Build Features using Domain Knowledge

> The best algorithms in the world cannot discover business logic that isn't physically present in the training matrix.

## What You Will Learn
- Transform institutional business rules explicitly into code
- Write vectorised operations to map continuous values conditionally

## Step 1: Flagging High-Value Interactions

A Retail dataset might possess hundreds of columns, but the business structurally defines a "VIP Customer" purely via an arbitrary threshold: someone who spent $> \$100$ and registered $> 5$ years ago. 

An algorithm will struggle blindly to discover this exact boundary linearly. You must explicitly inject this Domain Knowledge geometrically!

```python
import pandas as pd
import numpy as np
import seaborn as sns

# We utilize Pandas DataFrame manipulation without external libraries 
df = pd.DataFrame({
    'total_spend': [50, 150, 200, 45],
    'years_registered': [1, 6, 2, 8]
})

# Condition: Spend > 100 AND Years > 5
condition = (df['total_spend'] > 100) & (df['years_registered'] > 5)

# np.where executes this vectorised across 10 million rows instantly
df['is_vip'] = np.where(condition, 1, 0)

print(df)
```

??? example "Expected Output"
    ```text
       total_spend  years_registered  is_vip
    0           50                 1       0
    1          150                 6       1
    2          200                 2       0
    3           45                 8       0
    ```

## Step 2: Complex Business Logic (np.select)

Instead of a binary `1/0` VIP status dynamically, your company might have complex tiered pricing logically. 

Instead of writing dangerously slow `.apply(lambda x: ...)` functions natively, `np.select()` structurally chains multiple conditions simultaneously. 

```python
df_flights = sns.load_dataset('flights')

# We know historically logically that July/August are "Peak", 
# Dec/Jan are "Holiday", and everything else is "Off-Peak"

conditions = [
    df_flights['month'].isin(['Jul', 'Aug']),
    df_flights['month'].isin(['Dec', 'Jan'])
]

choices = ['Peak', 'Holiday']

# The 'default' is what happens geographically if no conditions trigger 
df_flights['seasonality_tier'] = np.select(conditions, choices, default='Off-Peak')

print(df_flights.sample(5, random_state=1))
```

??? example "Expected Output"
    ```text
         year month  passengers seasonality_tier
    94   1956   Nov         271         Off-Peak
    73   1955   Feb         233         Off-Peak
    55   1953   Aug         272             Peak
    115  1958   Aug         505             Peak
    67   1954   Aug         293             Peak
    ```

!!! tip "Workplace Tip"
    Whenever a domain expert logically tells you "Sales explicitly drop whenever it rains consecutively after a bank holiday", mechanically encode that EXACT sentence recursively into a mathematical feature `is_rain_after_holiday = [0,1]`. Domain experts understand causal reality natively better than XGBoost does experimentally!

## KSB Mapping

| KSB | Description | How This Guide Addresses It |
|-----|-------------|-------------------------------|
| S12 | Feature engineering | Mechanically translating verbal business thresholds functionally into tensor arrays |
| B2 | Logical and analytical approach | Explicitly leveraging enterprise structural knowledge programmatically |
