# SARIMA (Seasonal ARIMA)

> SARIMA extends ARIMA to handle seasonality by adding seasonal AR, differencing, and MA terms: SARIMA(\(p, d, q\))(\(P, D, Q, s\)).

## The Seasonal Parameters

| Parameter | Meaning |
|-----------|---------|
| \(P\) | Seasonal autoregressive order |
| \(D\) | Seasonal differencing order (usually 0 or 1) |
| \(Q\) | Seasonal moving average order |
| \(s\) | Length of the seasonal cycle (e.g., 12 for monthly data with yearly seasonality, 7 for daily data with weekly seasonality) |

## When to Use SARIMA vs ARIMA

- Use **ARIMA** when the series has a trend but no repeating seasonal pattern.
- Use **SARIMA** when the series has a clear seasonal cycle (e.g., monthly sales peaking every December).

## Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate synthetic monthly data with yearly seasonality
rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", periods=72, freq="MS")
trend = np.linspace(100, 160, 72)
seasonal = 15 * np.sin(2 * np.pi * np.arange(72) / 12)
noise = rng.normal(0, 3, 72)

ts = pd.Series(trend + seasonal + noise, index=dates, name="value")

# Fit SARIMA(1,1,1)(1,1,0,12)
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
fitted = model.fit(disp=False)
print(fitted.summary())

# Forecast 12 months
forecast = fitted.forecast(steps=12)

plt.figure(figsize=(10, 4))
ts.plot(label="Observed")
forecast.plot(label="Forecast", color="red", linestyle="--")
plt.legend()
plt.title("SARIMA(1,1,1)(1,1,0,12) Forecast")
plt.tight_layout()
plt.show()
```

## Model Selection

Use AIC (Akaike Information Criterion) to compare different orders:

```python
# Compare several model orders
best_aic = float("inf")
best_order = None

for p in range(3):
    for q in range(3):
        try:
            m = SARIMAX(ts, order=(p, 1, q), seasonal_order=(1, 1, 0, 12))
            r = m.fit(disp=False)
            if r.aic < best_aic:
                best_aic, best_order = r.aic, (p, 1, q)
        except Exception:
            continue

print(f"Best order: {best_order}, AIC: {best_aic:.2f}")
```

!!! tip "Workplace Tip"
    Use `pmdarima.auto_arima(seasonal=True, m=12)` for automated SARIMA order selection. This handles the grid search for you and is far more robust than manual iteration.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
