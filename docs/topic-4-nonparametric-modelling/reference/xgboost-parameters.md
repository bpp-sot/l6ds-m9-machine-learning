# XGBoost & LightGBM Parameters

> A quick reference for the most important hyperparameters to tune in XGBoost and LightGBM.

## Core Parameters

### XGBoost (`XGBClassifier` / `XGBRegressor`)
*   `n_estimators`: Number of boosting rounds (trees). Default: 100.
*   `learning_rate` (eta): Step size shrinkage used to prevent overfitting. Default: 0.3.
*   `max_depth`: Maximum depth of a tree. Default: 6.
*   `subsample`: Subsample ratio of the training instances. Default: 1.
*   `colsample_bytree`: Subsample ratio of columns when constructing each tree. Default: 1.

### LightGBM (`LGBMClassifier` / `LGBMRegressor`)
*   `n_estimators`: Number of boosting rounds. Default: 100.
*   `learning_rate`: Shrinkage rate. Default: 0.1.
*   `num_leaves`: Max number of leaves in one tree. Main parameter to control complexity. Default: 31.
*   `max_depth`: Limit the max depth. Default: -1 (no limit).
*   `subsample` (bagging_fraction): Like XGBoost.
*   `colsample_bytree` (feature_fraction): Like XGBoost.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
