# Why Stationarity Matters

> Most traditional time series forecasting models mathematically require the data to be stationary.

## The Core Concept

If a time series is not stationary, its statistical properties—like mean and variance—change over time. If the basic rules of the game keep changing, how can an algorithm predict the future?

**A non-stationary series:**
*   Trends upwards or downwards.
*   Has wilder swings (higher variance) in summer than in winter.
*   Has seasonal patterns that aren't constant.

**A stationary series:**
*   Idles roughly around a zero mean.
*   Has a constant variance (the wiggles are the same size everywhere).

By differencing (subtracting today's value from yesterday's), we remove the trend and often make the series stationary. We then forecast *the differences* and convert them back into real values.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
