# Facebook Prophet

> Prophet is an automated, additive forecasting framework developed by Meta that handles trends, seasonality, and holidays with minimal manual tuning.

## Why Prophet?

- Handles missing data and outliers gracefully.
- Automatically detects changepoints in the trend.
- Built-in support for daily, weekly, and yearly seasonality.
- Easy to add custom seasonalities and holiday effects.

## Implementation

```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create synthetic daily data with trend + weekly seasonality
dates = pd.date_range("2022-01-01", periods=365 * 2, freq="D")
rng = np.random.default_rng(42)
trend = np.linspace(50, 150, len(dates))
weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = rng.normal(0, 3, len(dates))

# Prophet requires columns named 'ds' (date) and 'y' (value)
df = pd.DataFrame({
    "ds": dates,
    "y": trend + weekly + noise
})

# 1. Fit the model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)

# 2. Create future dataframe and predict
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 3. Plot
model.plot(forecast)
```

## Component Plot

```python
# Visualise trend and seasonality components separately
model.plot_components(forecast)
```

## Adding Custom Features

```python
# Add holiday effects
holidays = pd.DataFrame({
    "holiday": "bank_holiday",
    "ds": pd.to_datetime(["2024-01-01", "2024-12-25", "2024-04-01"]),
    "lower_window": 0,
    "upper_window": 1
})

model = Prophet(holidays=holidays)
```

!!! tip "Workplace Tip"
    Prophet is excellent for rapid prototyping and business reporting. For maximum accuracy on complex patterns, consider gradient-boosting approaches (e.g., LightGBM with lag features) or `NeuralProphet`.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Using an automated forecasting framework for rapid time series prediction |
