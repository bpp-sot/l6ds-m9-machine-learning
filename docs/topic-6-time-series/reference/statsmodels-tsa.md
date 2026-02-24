# Statsmodels TSA Reference

> Quick lookup for the `statsmodels.tsa` API — the core library for classical time series analysis in Python.

## Key Classes

| Class | Import | Purpose |
|-------|--------|---------|
| `ARIMA` | `from statsmodels.tsa.arima.model import ARIMA` | Non-seasonal ARIMA models |
| `SARIMAX` | `from statsmodels.tsa.statespace.sarimax import SARIMAX` | Seasonal ARIMA with exogenous variables |
| `ExponentialSmoothing` | `from statsmodels.tsa.holtwinters import ExponentialSmoothing` | Holt-Winters exponential smoothing |
| `SimpleExpSmoothing` | `from statsmodels.tsa.holtwinters import SimpleExpSmoothing` | Single exponential smoothing (no trend/seasonality) |

## Key Functions

| Function | Import | Purpose |
|----------|--------|---------|
| `adfuller` | `from statsmodels.tsa.stattools import adfuller` | Augmented Dickey-Fuller test for stationarity |
| `seasonal_decompose` | `from statsmodels.tsa.seasonal import seasonal_decompose` | Decompose into trend, seasonal, residual |
| `plot_acf` | `from statsmodels.graphics.tsaplots import plot_acf` | Plot autocorrelation function |
| `plot_pacf` | `from statsmodels.graphics.tsaplots import plot_pacf` | Plot partial autocorrelation function |

## Common Workflow

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# 1. Create / load time series
dates = pd.date_range("2020-01-01", periods=120, freq="MS")
ts = pd.Series(np.random.default_rng(42).normal(0, 1, 120).cumsum(), index=dates)

# 2. Test for stationarity
result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

# 3. Difference if non-stationary (p > 0.05)
if result[1] > 0.05:
    ts_diff = ts.diff().dropna()
    print("Series differenced once.")

# 4. Fit ARIMA
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# 5. Forecast
forecast = fitted.forecast(steps=12)
print(forecast)
```

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Using classical statistical tools for time series modelling |
