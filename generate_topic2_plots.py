import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs('docs/assets/images', exist_ok=True)
sns.set_theme(style='whitegrid')

# ---------------------------------------------------------
# 1. Creating Features
# ---------------------------------------------------------
print("Generating plots for Creating Features...")
df_titanic = sns.load_dataset('titanic')

# Create a new feature 'family_size'
df_titanic['family_size'] = df_titanic['sibsp'] + df_titanic['parch'] + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df_titanic, x='sibsp', y='survived', ax=axes[0], errorbar=None, color='#6E368A')
axes[0].set_title('Survival by SibSp')

sns.barplot(data=df_titanic, x='family_size', y='survived', ax=axes[1], errorbar=None, color='#2D2D2D')
axes[1].set_title('Survival by Engineered Family Size')

plt.tight_layout()
plt.savefig('docs/assets/images/topic2-titanic-familysize.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 2. Datetime & Text Features
# ---------------------------------------------------------
print("Generating plots for Datetime Features...")
# Using a synthetic time series or flights dataset
df_flights = sns.load_dataset('flights')

# flights has 'year' and 'month'. Let's convert to datetime
df_flights['date'] = pd.to_datetime(df_flights['year'].astype(str) + '-' + df_flights['month'].astype(str) + '-01')
df_flights.set_index('date', inplace=True)

plt.figure(figsize=(12, 5))
sns.lineplot(data=df_flights, x=df_flights.index, y='passengers', color='#6E368A', linewidth=2)
plt.title('Monthly Flights Over Time (Raw Target)')
plt.tight_layout()
plt.savefig('docs/assets/images/topic2-flights-raw.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 3. Filter Methods (SelectKBest)
# ---------------------------------------------------------
print("Generating plots for Filter Methods...")
df_penguins = sns.load_dataset('penguins').dropna()
X = df_penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df_penguins['species']

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

scores_df = pd.DataFrame({'Feature': X.columns, 'F-Score': selector.scores_})
scores_df = scores_df.sort_values(by='F-Score', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=scores_df, x='F-Score', y='Feature', palette='viridis')
plt.title('ANOVA F-Value Feature Importance (Penguins dataset)')
plt.tight_layout()
plt.savefig('docs/assets/images/topic2-filter-anova.png', dpi=150, facecolor='white')
plt.close()


# ---------------------------------------------------------
# 4. Embedded Methods (Random Forest Feature Importance)
# ---------------------------------------------------------
print("Generating plots for Embedded Methods...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', color='#D94D26') # Some accent color
plt.title('Random Forest Gini Importance')
plt.tight_layout()
plt.savefig('docs/assets/images/topic2-embedded-rf.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 5. PCA Dimensionality Reduction
# ---------------------------------------------------------
print("Generating plots for PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cumulative Explained Variance
axes[0].plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='#6E368A')
axes[0].set_xlabel('Number of Principal Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Scree Plot')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].axhline(y=0.90, color='r', linestyle='--', label='90% Variance Threshold')
axes[0].legend()

# Plot 2: 2D Projection
X_pca = pca.transform(X_scaled)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=axes[1], palette='Set2')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].set_title('2D PCA Projection of Penguins')

plt.tight_layout()
plt.savefig('docs/assets/images/topic2-pca.png', dpi=150, facecolor='white')
plt.close()

print("All Topic 2 plots generated successfully.")
