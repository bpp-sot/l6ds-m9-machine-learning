# How to Decompose a Time Series

> Decomposing a time series lets you visualise its core components: **Trend**, **Seasonality**, and **Residuals**.

## Why Decompose?

Decomposition separates the observed signal into interpretable parts:

| Component | What It Captures |
|-----------|-----------------|
| **Trend** | Long-term direction (upward, downward, flat) |
| **Seasonality** | Repeating patterns at fixed intervals (daily, weekly, yearly) |
| **Residuals** | Random noise left after removing trend and seasonality |

Understanding these components helps you choose the right forecasting model and diagnose problems (e.g., a strong seasonal pattern suggests SARIMA over ARIMA).

## Additive vs Multiplicative

- **Additive:** `Observed = Trend + Seasonal + Residual` — use when seasonal amplitude is constant over time.
- **Multiplicative:** `Observed = Trend × Seasonal × Residual` — use when seasonal amplitude grows with the trend.

## Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Create synthetic monthly data with trend and seasonality
dates = pd.date_range(start="2020-01-01", periods=72, freq="MS")
trend = np.linspace(100, 200, 72)
season = 10 * np.sin(np.linspace(0, 12 * np.pi, 72))
noise = np.random.default_rng(42).normal(0, 3, 72)

ts = pd.Series(trend + season + noise, index=dates)

# Decompose — period=12 for monthly data with yearly seasonality
result = seasonal_decompose(ts, model="additive", period=12)
result.plot()
plt.tight_layout()
plt.show()
```

!!! tip "Workplace Tip"
    If the seasonal swings grow proportionally with the level of the series, switch to `model='multiplicative'`. A quick visual check of the raw plot usually makes this obvious.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Decomposing time series to inform model selection |
