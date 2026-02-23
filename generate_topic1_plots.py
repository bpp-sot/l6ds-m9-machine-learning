import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Ensure images directory exists
os.makedirs('docs/assets/images', exist_ok=True)

# Set base style
sns.set_theme(style='whitegrid')

# ---------------------------------------------------------
# 1. Loading & Exploring Data
# ---------------------------------------------------------
print("Generating plots for Loading & Exploring Data...")
df_penguins = sns.load_dataset('penguins')

# Missingno matrix
msno.matrix(df_penguins, figsize=(10, 5))
plt.title('Penguins Dataset: Missing Value Patterns', fontsize=16)
plt.tight_layout()
plt.savefig('docs/assets/images/topic1-penguins-msno-matrix.png', dpi=150, facecolor='white')
plt.close()

# Distribution analysis (numeric)
numeric_cols = df_penguins.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data=df_penguins, x=col, kde=True, ax=axes[i], color='#6E368A')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('docs/assets/images/topic1-penguins-distributions.png', dpi=150, facecolor='white')
plt.close()

# Correlation matrix
plt.figure(figsize=(8, 6))
corr_matrix = df_penguins.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('docs/assets/images/topic1-penguins-correlation.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 2. Handling Missing Values
# ---------------------------------------------------------
print("Generating plots for Handling Missing Values...")
df_titanic = sns.load_dataset('titanic')

plt.figure(figsize=(10, 5))
sns.histplot(data=df_titanic, x='age', bins=30, kde=True, color='skyblue', label='Original (missing removed)')
# Simple mean imputation for visualization
df_imputed = df_titanic.copy()
df_imputed['age'] = df_imputed['age'].fillna(df_imputed['age'].mean())
sns.histplot(data=df_imputed, x='age', bins=30, kde=True, color='red', alpha=0.3, label='Mean Imputed')
plt.title('Impact of Mean Imputation on Age Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/images/topic1-titanic-imputation-impact.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 3. Scaling & Normalisation
# ---------------------------------------------------------
print("Generating plots for Scaling & Normalisation...")
df_diamonds = sns.load_dataset('diamonds').sample(1000, random_state=42) # sample for clear plotter

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.scatterplot(data=df_diamonds, x='carat', y='price', alpha=0.5, ax=axes[0])
axes[0].set_title('Original Scale')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df_diamonds[['carat', 'price']]), columns=['carat', 'price'])
sns.scatterplot(data=df_std, x='carat', y='price', alpha=0.5, ax=axes[1], color='green')
axes[1].set_title('StandardScaler (Z-Score)')

scaler_minmax = MinMaxScaler()
df_mm = pd.DataFrame(scaler_minmax.fit_transform(df_diamonds[['carat', 'price']]), columns=['carat', 'price'])
sns.scatterplot(data=df_mm, x='carat', y='price', alpha=0.5, ax=axes[2], color='purple')
axes[2].set_title('MinMaxScaler (0 to 1)')

plt.tight_layout()
plt.savefig('docs/assets/images/topic1-diamonds-scaling.png', dpi=150, facecolor='white')
plt.close()

# ---------------------------------------------------------
# 4. Outliers
# ---------------------------------------------------------
print("Generating plots for Outliers...")
# Add some synthetic outliers to diamonds dataset
df_outliers = df_diamonds.copy()
df_outliers.loc[df_outliers.index[0], 'price'] = 25000
df_outliers.loc[df_outliers.index[1], 'carat'] = 6.0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df_diamonds, y='price', x='cut', ax=axes[0], palette='viridis')
axes[0].set_title('Box Plot for Outlier Detection')

sns.scatterplot(data=df_diamonds, x='carat', y='price', alpha=0.5, ax=axes[1])
# Highlight the IQR bounds for carat
Q1 = df_diamonds['carat'].quantile(0.25)
Q3 = df_diamonds['carat'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
axes[1].axvline(upper_bound, color='red', linestyle='--', label=f'Upper IQR Bound ({upper_bound:.2f})')
axes[1].legend()
axes[1].set_title('Scatter Plot IQR Bound')

plt.tight_layout()
plt.savefig('docs/assets/images/topic1-diamonds-outliers.png', dpi=150, facecolor='white')
plt.close()

print("All plots generated successfully.")
