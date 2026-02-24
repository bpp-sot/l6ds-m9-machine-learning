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
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
