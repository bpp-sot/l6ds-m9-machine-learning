# ML vs Statistical Forecasting

> You can use XGBoost for time series, but it's fundamentally different from ARIMA.

## Statistical Models (ARIMA, SARIMA)
*   **How they work:** They explicitly mode time, autocorrelation (lags), and seasonality.
*   **Pros:** Highly interpretable, very strong on small datasets, built-in confidence intervals.
*   **Cons:** Require strict assumptions (stationarity), struggle with many exogenous (external) variables, can't naturally train across multiple different time series at once.

## Machine Learning Models (XGBoost, LSTMs)
*   **How they work:** You have to extract time features (e.g., "is_weekend", "month_number", "lag_1", "lag_7") and feed them in as standard tabular machine learning.
*   **Pros:** Can easily consume hundreds of external features (weather, price, holidays), often win mapping non-linear combinations of features.
*   **Cons:** No built-in understanding of time (they just see rows of data), they cannot extrapolate trends (a tree can never predict a value higher than it saw in training).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | ARIMA, SARIMA, and exponential smoothing foundations |
| K4.2 | Predictive analytics and ML techniques | Time series forecasting and model comparison |
| K5.3 | Common patterns in real-world data | Identifying trends, seasonality, and stationarity |
| S1 | Scientific methods and hypothesis testing | Stationarity testing, model diagnostics, forecast validation |
| S4 | Analysis and models to inform outcomes | Building forecasts to support business planning |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of forecast accuracy and limitations |
