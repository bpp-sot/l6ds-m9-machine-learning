# How-to: Create Domain-Specific Features

## The Problem
Sometimes, standard mathematical logic (like automated polynomial expansion) isn't enough. Models lack human context. If a user buys a snow shovel in July, that implies something entirely different than buying one in December. 

## The Solution
You must create custom function transformations mapped explicitly to **Domain Knowledge** (the logic of the physical industry).

```python
import pandas as pd

# Retail transactional data
retail = pd.DataFrame({
    'Customer_Age': [22, 45, 17, 65],
    'Purchase_Date': pd.to_datetime(['2023-12-15', '2023-07-20', '2023-11-25', '2023-03-10']),
    'Item_Category': ['Winter_Coat', 'Winter_Coat', 'Video_Games', 'Gardening']
})

# 1. Generational Binning (Domain: Marketing Cohorts)
def map_generation(age):
    if age < 25: return 'Gen_Z'
    if age < 41: return 'Millennial'
    if age < 57: return 'Gen_X'
    return 'Boomer'

retail['Demographic_Cohort'] = retail['Customer_Age'].apply(map_generation)

# 2. Seasonality Flags (Domain: Retail Cycles)
def is_holiday_season(date_obj):
    # Flag purchases made in November or December
    return 1 if date_obj.month in [11, 12] else 0

retail['Is_Holiday_Rush'] = retail['Purchase_Date'].apply(is_holiday_season)

# 3. Contextual Anomalies (Domain: Inventory Fraud or Clearance Sales)
# Buying a Winter Coat in July is highly anomalous. Let's flag it.
retail['Is_Off_Season_Purchase'] = (
    (retail['Purchase_Date'].dt.month.isin([6, 7, 8])) & 
    (retail['Item_Category'] == 'Winter_Coat')
).astype(int)

print(retail[['Item_Category', 'Purchase_Date', 'Is_Off_Season_Purchase']])
```

## Discussion

### Why Domain Knowledge Wins
In competitions like Kaggle, the winning parameter is almost rarely a hyper-optimized Learning Rate. It is always a uniquely sculpted feature developed by someone who deeply understands the dataset's physical context.

### Applying to the Assessment
During your L6 apprenticeship presentation, if you demonstrate a feature engineered explicitly because you interviewed a colleague and learned a business quirk, you will immediately score high marks in **Business Acumen** and **Communication**.
