# Scikit-Learn Preprocessing Reference

> The `sklearn.preprocessing` module is the engine driving mathematical transformations.

## Encoders (Categorical to Numeric)

Mapping unstructured text structurally into arrays.

| Class | Use Case | Example |
|---|---|---|
| `OrdinalEncoder()` | **Hierarchical/Ranked Text**: Low < Medium < High | `[[0], [1], [2]]` |
| `OneHotEncoder(drop='first')` | **Nominal Text**: Red vs Blue vs Green | `[[1,0], [0,1], [0,0]]` |
| `LabelEncoder()` | **Target Variables strictly (`Y`)**: Binary or Multiclass strings | `0, 1` |

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Ordinal mathematically enforces scale
rank_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])

# OneHot purely routes string values out to parallel binary columns
ohe = OneHotEncoder(sparse_output=False, drop='first')
```

## Scalers (Numeric to Numeric)

Harmonising the distance between numerical ranges.

| Class | Behaviour | Use Case |
|---|---|---|
| `StandardScaler()` | Z-score scaling. Mean is strictly 0, std is 1. | **DEFAULT**: Linear regressions, SVMs, PCA |
| `MinMaxScaler()` | Compresses purely between minimum 0.0 to maximum 1.0 | Deep Learning Tensors (Neural Networks) |
| `RobustScaler()` | Scales purely around Median and strictly Interquartile Range | Data containing severe extreme outliers |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sets feature distribution to Mean = 0, Std = 1
std = StandardScaler()

# Sets boundary dynamically between Min = 0.0, Max = 1.0 
minmax = MinMaxScaler()
```

## Imputers (Missing Data)

Located uniquely in `sklearn.impute`.

| Class | Strategy | Use Case |
|---|---|---|
| `SimpleImputer(strategy='mean')` | Fills statically with statistical Average | Normally distributed continuous algorithms |
| `SimpleImputer(strategy='median')`| Fills statically with geometric middle | Highly skewed geometric boundaries |
| `SimpleImputer(strategy='most_frequent')` | Fills statically with textual Mode | Pure categorical (Text/Object) columns |

```python
from sklearn.impute import SimpleImputer

# Instantiation mapping structurally to Median arithmetic
imputer = SimpleImputer(strategy='median')
```

!!! info "Assessment Connection"
    For your K3 (Data Management) portfolio, documenting why `StandardScaler` was chosen explicitly over `MinMaxScaler` based on your analysis of the statistical feature skew guarantees top marks from assessment moderators.
