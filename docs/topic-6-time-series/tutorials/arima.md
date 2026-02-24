# ARIMA Models

> AutoRegressive Integrated Moving Average (ARIMA) combines three components to model non-seasonal time series: autoregression (AR), differencing (I), and moving average (MA).

## The Three Components

| Component | Parameter | What It Does |
|-----------|-----------|--------------|
| **AR (AutoRegressive)** | \(p\) | Uses past values to predict the current value |
| **I (Integrated)** | \(d\) | Number of times the series is differenced to achieve stationarity |
| **MA (Moving Average)** | \(q\) | Uses past forecast errors to predict the current value |

The model is specified as **ARIMA(\(p, d, q\))**.

## Choosing \(p, d, q\)

1. **\(d\):** Difference the series until it is stationary (ADF test p-value < 0.05). Usually \(d = 0, 1,\) or \(2\).
2. **\(p\):** Look at the PACF plot — the lag where it cuts off.
3. **\(q\):** Look at the ACF plot — the lag where it cuts off.

## Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Generate synthetic non-stationary data
rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=120, freq="MS")
ts = pd.Series(rng.normal(0, 1, 120).cumsum() + np.linspace(0, 20, 120), index=dates)

# 1. Check stationarity
adf_result = adfuller(ts)
print(f"ADF p-value: {adf_result[1]:.4f}")

# 2. Fit ARIMA(1,1,1)
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# 3. Forecast 12 months ahead
forecast = fitted.forecast(steps=12)

plt.figure(figsize=(10, 4))
ts.plot(label="Observed")
forecast.plot(label="Forecast", color="red", linestyle="--")
plt.legend()
plt.title("ARIMA(1,1,1) Forecast")
plt.tight_layout()
plt.show()
```

## Model Diagnostics

```python
# Check residuals — they should look like white noise
fitted.plot_diagnostics(figsize=(10, 6))
plt.tight_layout()
plt.show()
```

!!! tip "Workplace Tip"
    Use `pmdarima.auto_arima()` to automatically search for the best (\(p, d, q\)) combination using AIC. This saves manual inspection of ACF/PACF plots.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
