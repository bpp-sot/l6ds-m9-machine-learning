# Stationarity & Differencing

> Most statistical time series models (ARIMA, SARIMA) require **stationarity** — meaning the mean, variance, and autocorrelation structure do not change over time.

## Why Stationarity Matters

Non-stationary data has trends, changing variance, or seasonal shifts that violate the assumptions of ARIMA-type models. Fitting these models on non-stationary data produces unreliable forecasts.

## Testing for Stationarity

### Augmented Dickey-Fuller (ADF) Test

- **Null hypothesis:** The series has a unit root (non-stationary).
- **If p-value < 0.05:** Reject the null → the series is stationary.
- **If p-value ≥ 0.05:** Fail to reject → the series is non-stationary (needs differencing).

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Non-stationary: random walk with drift
rng = np.random.default_rng(42)
ts = pd.Series(rng.normal(0, 1, 200).cumsum())

result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# Expected: p-value > 0.05 → non-stationary
```

## Differencing

Differencing subtracts each observation from its predecessor, removing trends:

```python
# First difference — removes linear trend
ts_diff1 = ts.diff().dropna()

# Second difference — removes quadratic trend (rarely needed)
ts_diff2 = ts_diff1.diff().dropna()

# Test again
result2 = adfuller(ts_diff1)
print(f"After differencing — p-value: {result2[1]:.4f}")
```

## Visual Check

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ts.plot(ax=axes[0], title="Original (Non-Stationary)")
ts_diff1.plot(ax=axes[1], title="First Difference (Stationary)")
plt.tight_layout()
plt.show()
```

!!! warning "Common Pitfall"
    Do not over-difference. If one round of differencing makes the series stationary (\(d = 1\)), stop. Over-differencing introduces artificial patterns and degrades model performance.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Preparing time series data for statistical modelling |
