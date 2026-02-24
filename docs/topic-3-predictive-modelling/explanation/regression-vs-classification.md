# Regression vs Classification

> Every supervised ML problem falls into one of two categories — the distinction is determined entirely by the type of your target variable.

## Classification

Your target variable is **categorical** — it belongs to a discrete set of classes.

| Example | Target | Classes |
|---------|--------|---------|
| Spam detection | `is_spam` | 0 (No), 1 (Yes) |
| Disease diagnosis | `condition` | "Healthy", "Diabetic", "Pre-diabetic" |
| Customer segment | `tier` | "Gold", "Silver", "Bronze" |

**Algorithms:** `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Regression

Your target variable is **continuous** — it can take any numeric value within a range.

| Example | Target | Range |
|---------|--------|-------|
| House pricing | `sale_price` | £50,000 – £2,000,000 |
| Temperature forecasting | `temp_celsius` | -30.0 – 50.0 |
| Revenue prediction | `monthly_revenue` | £0 – £10,000,000 |

**Algorithms:** `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`

**Metrics:** MAE, MSE, RMSE, R²

## The Decision Rule

Ask yourself one question: **"Can I meaningfully average two target values?"**

- If averaging makes sense (e.g., the average of £200k and £300k is £250k), it is **regression**.
- If averaging is nonsensical (e.g., the average of "Cat" and "Dog" is meaningless), it is **classification**.

!!! warning "Common Pitfall"
    Binary targets encoded as 0/1 are still classification, not regression — even though they look numeric. Use `LogisticRegression`, not `LinearRegression`.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
