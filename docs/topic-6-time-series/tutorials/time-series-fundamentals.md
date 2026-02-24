# Time Series Fundamentals

> A time series is data ordered by time. Analysing it means understanding the past to predict the future.

## What Makes Time Series Special?

Unlike tabular ML data, time series has a **temporal dependency** — the order of observations matters. You cannot shuffle rows without destroying information.

## Core Components

Every time series can be thought of as a combination of:

| Component | Description | Example |
|-----------|-------------|---------|
| **Trend** | Long-term upward or downward movement | Increasing global temperatures |
| **Seasonality** | Repeating pattern at fixed intervals | Ice cream sales peaking every summer |
| **Noise** | Random, unpredictable variation | Daily stock price fluctuations |

## Creating a Time Series in Pandas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a date range as the index
dates = pd.date_range(start="2024-01-01", periods=365, freq="D")

# Simulate: trend + seasonality + noise
rng = np.random.default_rng(42)
trend = np.linspace(100, 150, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = rng.normal(0, 3, 365)

ts = pd.Series(trend + seasonality + noise, index=dates, name="value")

ts.plot(figsize=(10, 4), title="Synthetic Time Series")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
```

## Key Pandas Operations

```python
# Resample to monthly average
monthly = ts.resample("MS").mean()

# Rolling 7-day moving average
rolling = ts.rolling(window=7).mean()

# Shift (lag) the series
lagged = ts.shift(1)  # Previous day's value
```

## Important Rules

1. **Never shuffle** time series data.
2. **Respect temporal order** when splitting into train/test sets.
3. **Always set the date as the index** and ensure it is a `DatetimeIndex`.
4. **Check for missing timestamps** — gaps break most models.

!!! tip "Workplace Tip"
    Always plot your time series first. A simple line chart reveals trends, seasonality, and outliers faster than any statistical test.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
