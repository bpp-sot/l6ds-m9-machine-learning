# How to Generate a Feature Importance Report

> Building an algorithm is Science. Convincing your Director to trust it is explicitly Communication. 

## What You Will Learn
- Extract native coefficients securely from a trained ML model
- Construct a structurally clean DataFrame dynamically for reporting
- Plot findings explicitly using Seaborn functionally

## Step 1: Training the Reporter

A Random Forest dynamically evaluates internal Gini impurity logically at every specific decision node mathematically. We can extract these computations globally!

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = sns.load_dataset('penguins').dropna()
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# Train globally
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
```

## Step 2: Extracting and Formatting

The `rf.feature_importances_` array natively is just raw floats. We must architecturally bind them cleanly back computationally to the `X.columns` titles!

```python
# Construct the explicit reporting table 
report = pd.DataFrame({
    'Feature': X.columns,
    'Gini_Importance': rf.feature_importances_
})

# Mathematically sort natively and mechanically compute percentages
report = report.sort_values(by='Gini_Importance', ascending=False)
report['Percentage'] = (report['Gini_Importance'] * 100).round(2).astype(str) + '%'

print(report[['Feature', 'Percentage']])
```

??? example "Expected Output"
    ```text
                 Feature Percentage
    2  flipper_length_mm      41.9%
    0     bill_length_mm     39.57%
    1      bill_depth_mm     12.64%
    3        body_mass_g      5.88%
    ```

## Step 3: Visualising the Report

Executives do not read tables. They strictly scan visuals rapidly!

```python
plt.figure(figsize=(10, 6))

# Plot structurally
ax = sns.barplot(data=report, x='Gini_Importance', y='Feature', color='#2D2D2D')

# Dynamically annotate the strict values computationally directly onto the bars
for i, v in enumerate(report['Gini_Importance']):
    ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center')

plt.title('Predictive Drivers for Penguin Species Classification', fontsize=14)
plt.xlabel('Algorithm Gini Contribution Score')
plt.xlim(0, 0.5) 
plt.box(False) # Remove chart borders natively to look clean and professional
plt.show()
```

??? example "Expected Output"
    *(A cleanly formatted Seaborn bar chart displaying Flipper Length and Bill Length heavily dominating the analytical outcome visually.)*

!!! tip "Workplace Tip"
    In your EPA explicitly (and natively in PowerPoint meetings), always structure Feature Importance specifically as "Predictive Drivers". Do not say "Here are my Gini Coefficients natively." Explicitly present "These are the 4 behaviors that dynamically drive 90% of our Churn logically!"

## KSB Mapping

| KSB | Description | How This Guide Addresses It |
|-----|-------------|-------------------------------|
| S16 | Communication | Visually presenting algorithm internals natively mathematically to technical and non-technical structurally |
| K6 | Data analytics and visualisation | Designing annotated structural Seaborn bar plots efficiently |
