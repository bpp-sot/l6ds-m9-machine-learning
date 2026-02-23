# Reference: Sklearn Preprocessing API

This page acts as a quick-lookup repository for Scikit-Learn's `sklearn.preprocessing` library classes. 

## The Standard Transformer API

All valid transformers within Scikit-Learn follow a tri-method design paradigm:

| Method | Parameters | Action |
|--------|------------|--------|
| `fit()` | `(X, y=None)` | Learns the required parameters from the data structure (e.g., mean, standard deviation, max value, unique classes). |
| `transform()` | `(X)` | Applies the learned transformation from `fit()` to the provided input matrix. |
| `fit_transform()` | `(X, y=None)`| Fits and transforms in a single, perfectly optimized computational step. **Only use this on Training Data.** |

> [!WARNING]
> **Data Leakage Risk:** Using `.fit_transform()` on `X_test` will overwrite the scaling logic learned from the training loop, causing a silent evaluation failure. Always utilize `.fit_transform()` on Training objects, and **strictly** `.transform()` on Validation/Testing vectors. 

---

## 1. Feature Scalars

All scalars modify numeric ranges to protect continuous gradient calculations without sacrificing structural topology.

### `StandardScaler()`
- **Logic**: Centers mean at 0 and scales to internal variance 1.
- **When to Use**: Normally distributed variables. Default choice.

### `MinMaxScaler(feature_range=(0,1))`
- **Logic**: Squashes limits to strict, static bounds.
- **When to Use**: Deep Learning pixels or hard range strictures. Highly influenced by outliers.

### `RobustScaler()`
- **Logic**: Removes Median and scales according to IQR range limits (1st quartile and 3rd quartile). 
- **When to Use**: Fields devastated by immense, irremovable outlier artifacts.

---

## 2. Categorical Encoders

Text encoding forces alphabetic patterns into machine-readable numeric formats.

### `OneHotEncoder(drop='first', sparse_output=False)`
- **Logic**: Projects each category class into a distinct Boolean feature column.
- **When to Use**: **Nominal** Categories.
- *Notes:* Setting `drop='first'` prevents multi-collinearity issues against non-regularized generalized linear engines.

### `OrdinalEncoder()`
- **Logic**: Enumerates classes progressively (0, 1, 2...).
- **When to Use**: Explicit **Ordinal** Categories where logic scales progressively (Low, Medium, High). 

### `TargetEncoder(smooth="auto")`
- **Logic**: Injects the historical mathematical Mean of the prediction target aligned specifically with the category group.
- **When to Use**: High Cardinality String lists (ZIP Code, VIN Num, Store Num).

---

## 3. Discretization

### `KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')`
- **Logic**: Partitions massive continuous data values into arbitrary categorical block definitions.
- **When to Use**: Mapping age ranges natively (20-30, 30-40, etc) to generalize tree relationships natively. Strategy variables can force identical bin sizes or identically sized population distributions inside the bins.
