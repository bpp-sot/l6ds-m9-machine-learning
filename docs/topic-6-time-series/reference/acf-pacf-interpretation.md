# ACF/PACF Interpretation Guide

> How to read Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots to determine the order of ARIMA models.

## What They Show

- **ACF (Autocorrelation Function):** Correlation between the series and its lagged values. Includes indirect effects through intermediate lags.
- **PACF (Partial Autocorrelation Function):** Correlation between the series and a specific lag, *after removing* the effects of all shorter lags.

## Reading the Plots

### Identifying AR Order (p) — Use PACF

If the PACF shows a **sharp cutoff** after lag \(p\) (significant spikes then drops to zero), the series has an AR(\(p\)) component.

### Identifying MA Order (q) — Use ACF

If the ACF shows a **sharp cutoff** after lag \(q\) (significant spikes then drops to zero), the series has an MA(\(q\)) component.

### Summary Table

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| Tails off slowly | Sharp cutoff at lag \(p\) | — | AR(\(p\)) |
| Sharp cutoff at lag \(q\) | — | Tails off slowly | MA(\(q\)) |
| Tails off slowly | Tails off slowly | — | ARMA(\(p, q\)) |

## Plotting ACF and PACF

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate synthetic AR(2) data
rng = np.random.default_rng(42)
n = 200
data = np.zeros(n)
for t in range(2, n):
    data[t] = 0.5 * data[t-1] + 0.3 * data[t-2] + rng.normal()

ts = pd.Series(data)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, lags=20, ax=axes[0])
plot_pacf(ts, lags=20, ax=axes[1])
plt.tight_layout()
plt.show()
```

In this example, the PACF should show significant spikes at lags 1 and 2, then cut off — confirming AR(2).

!!! tip "Workplace Tip"
    If you are unsure about the order, use `auto_arima` from the `pmdarima` library to automatically select (p, d, q) via AIC/BIC criteria.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
