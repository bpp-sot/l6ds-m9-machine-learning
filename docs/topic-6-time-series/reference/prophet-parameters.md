# Prophet Parameters

> Key hyperparameters for tuning Facebook Prophet.

## Core Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `growth` | `'linear'` | `'linear'` for linear trend, `'logistic'` for saturating growth (requires `cap` column) |
| `changepoint_prior_scale` | `0.05` | Controls trend flexibility. Higher values → more changepoints → risk of overfitting the trend |
| `seasonality_prior_scale` | `10` | Controls seasonality flexibility. Higher values → more aggressive seasonal fitting |
| `holidays_prior_scale` | `10` | Controls holiday effect flexibility |
| `seasonality_mode` | `'additive'` | `'additive'` or `'multiplicative'` — use multiplicative when seasonal amplitude grows with the trend |

## Changepoints

Prophet automatically detects trend changepoints (points where the growth rate shifts). You can control this with:

- `n_changepoints`: Number of potential changepoints (default 25).
- `changepoint_range`: Proportion of the series in which changepoints are placed (default 0.8 — last 20% is excluded to avoid overfitting the tail).

## Custom Seasonality

```python
from prophet import Prophet

model = Prophet()

# Add custom seasonality (e.g., monthly with 30.5 day period)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
```

## Tuning Example

```python
from prophet import Prophet
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D")
y = np.sin(np.linspace(0, 6 * np.pi, len(dates))) * 10 + np.linspace(50, 100, len(dates))
df = pd.DataFrame({"ds": dates, "y": y + np.random.default_rng(42).normal(0, 2, len(dates))})

model = Prophet(
    changepoint_prior_scale=0.1,   # More flexible trend
    seasonality_prior_scale=5,     # Moderate seasonality
    seasonality_mode="additive"
)
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
model.plot(forecast)
```

!!! tip "Workplace Tip"
    Start with default parameters. If the trend looks too rigid, increase `changepoint_prior_scale`. If seasonality looks noisy, decrease `seasonality_prior_scale`.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
