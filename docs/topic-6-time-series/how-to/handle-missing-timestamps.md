# Handle Missing Timestamps

> Missing data in time series creates uneven intervals, which breaks most forecasting models that assume regular spacing.

## The Problem

Real-world time series often have gaps — missing days, weekends, or sensor outages. Models like ARIMA and Prophet expect evenly spaced observations.

## Strategy

1. **Resample** to the expected frequency to create a complete date index.
2. **Fill** the gaps using an appropriate method.

## Implementation

```python
import pandas as pd
import numpy as np

# Create a time series with missing dates
dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05",
                         "2024-01-06", "2024-01-09", "2024-01-10"])
values = [100, 102, 108, 110, 115, 117]
ts = pd.Series(values, index=dates)

print("Before — gaps present:")
print(ts)

# 1. Resample to daily frequency (creates NaN for missing days)
ts_resampled = ts.resample("D").asfreq()

# 2. Forward fill — carry the last known value forward
ts_ffill = ts_resampled.ffill()

# OR: Linear interpolation — estimate between known points
ts_interp = ts_resampled.interpolate(method="linear")

print("\nForward Fill:")
print(ts_ffill)

print("\nLinear Interpolation:")
print(ts_interp)
```

## Choosing a Fill Method

| Method | When to Use |
|--------|-------------|
| `ffill()` | Sensor data where the last reading carries forward (e.g., temperature) |
| `bfill()` | When you expect the next value to be more representative |
| `interpolate(method='linear')` | When a smooth transition between points is reasonable |
| `interpolate(method='time')` | When gaps are irregular and you want time-weighted interpolation |

!!! warning "Common Pitfall"
    Do not forward-fill over very long gaps. If a sensor was offline for a week, carrying the last reading forward for 7 days introduces misleading flat segments. Consider dropping those periods or flagging them instead.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Handling missing data specific to time series analysis |
