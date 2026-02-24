# How to Evaluate Forecast Accuracy

> Time series forecasting requires specific metrics — like MAE, RMSE, and MAPE — calculated on a chronologically held-out test set.

## Golden Rule

**Never shuffle time series data.** The train/test split must respect temporal order: train on the past, test on the future.

## Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean of \|actual − predicted\| | Average absolute error in the original units |
| **RMSE** | √Mean of (actual − predicted)² | Penalises large errors more heavily than MAE |
| **MAPE** | Mean of \|actual − predicted\| / \|actual\| × 100 | Percentage error — useful for comparing across scales |

## Implementation

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Simulate actual vs predicted
rng = np.random.default_rng(42)
dates = pd.date_range("2024-01-01", periods=30, freq="D")
y_true = 100 + rng.normal(0, 5, 30).cumsum()
y_pred = y_true + rng.normal(0, 3, 30)  # Predictions with noise

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```

## Time Series Train/Test Split

```python
# Chronological split — no shuffling!
train = ts[:int(len(ts) * 0.8)]
test = ts[int(len(ts) * 0.8):]
```

!!! warning "Common Pitfall"
    Using `train_test_split(shuffle=True)` on time series data causes **data leakage** — you train on future data and test on past data, producing artificially inflated scores.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
