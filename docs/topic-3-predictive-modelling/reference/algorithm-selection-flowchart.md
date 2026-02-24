# Algorithm Selection Flowchart

> Choosing the right algorithm requires answering a short sequence of questions about your data and target variable.

## The Decision Flow

### Step 1: Do you have labels (a target variable `y`)?

- **No → Unsupervised Learning** (see Topic 5: Clustering — K-Means, DBSCAN).
- **Yes → Supervised Learning.** Move to Step 2.

### Step 2: Is your target variable continuous (e.g., £ price, temperature)?

- **Yes → Regression.**
    - *Start with:* `LinearRegression` (interpretable baseline).
    - *If non-linear patterns exist:* `RandomForestRegressor` or `GradientBoostingRegressor`.
    - *If you need regularisation:* `Ridge`, `Lasso`, or `ElasticNet`.
- **No → Classification.** Move to Step 3.

### Step 3: Classification — how complex is the decision boundary?

- *Start with:* `LogisticRegression` (interpretable baseline).
- *If non-linear patterns exist:* `RandomForestClassifier` or `GradientBoostingClassifier`.
- *If you need probability outputs:* Ensure the model supports `predict_proba()` (most tree-based and logistic models do; default SVM does not).
- *If you have very high-dimensional sparse data (e.g., text):* `MultinomialNB` or `SGDClassifier`.

### Step 4: Scale and performance considerations

| Consideration | Recommended Action |
|---------------|-------------------|
| Dataset > 100k rows | Use `HistGradientBoostingClassifier` or LightGBM for speed |
| Interpretability required | Use `LogisticRegression` or `DecisionTreeClassifier` (shallow) |
| Maximum accuracy needed | Use `GradientBoostingClassifier` / XGBoost with hyperparameter tuning |
| Many irrelevant features | Apply `Lasso` (L1) for automatic feature selection |

!!! tip "Workplace Tip"
    When in doubt, train a `RandomForestClassifier` with default parameters as your first model. It handles mixed feature types, non-linear relationships, and missing values (in some implementations) with minimal preprocessing.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S13 | Apply ML algorithms | Systematic algorithm selection based on problem characteristics |
