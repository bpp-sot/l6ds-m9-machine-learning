# Assessment Checklist

> Before you submit your M9 final assessment, check every box below to ensure you have covered the full ML workflow.

## The MVP Submission

- [ ] Have I formulated a clear ML business problem?
- [ ] Did I acquire, clean, and engineer features appropriately?
- [ ] Have I trained at least one baseline and one advanced model?
- [ ] Is there rigorous validation (CV, training vs test analysis)?
- [ ] Have I explicitly translated the model metric (Accuracy/RMSE) into a business metric (£ saved or time reduced)?
- [ ] Did I communicate the results effectively with a visualisation?

## Data Preparation

- [ ] Missing values handled (dropped, imputed, or flagged)
- [ ] Categorical features encoded (One-Hot, Ordinal, or Target Encoding)
- [ ] Numerical features scaled where required (StandardScaler for distance-based models)
- [ ] No data leakage — preprocessing fitted on training data only

## Modelling

- [ ] At least two algorithms compared (e.g., Logistic Regression vs Random Forest)
- [ ] Hyperparameters tuned systematically (GridSearch, RandomSearch, or Optuna)
- [ ] Cross-validation used — not a single train/test split
- [ ] Overfitting checked (training vs test score gap)

## Evaluation

- [ ] Appropriate metric chosen (not just accuracy — consider F1, ROC AUC, RMSE)
- [ ] Confusion matrix or classification report included (for classification)
- [ ] Residual analysis included (for regression)
- [ ] Confidence intervals or standard deviation reported

## Communication

- [ ] Feature importance visualised (SHAP, permutation, or tree-based)
- [ ] Results presented in plain English for a non-technical audience
- [ ] Business impact quantified (ROI, cost savings, efficiency gains)
- [ ] Limitations and next steps acknowledged

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K1 | Context of Data Science | Understanding where ML sits within the broader discipline |
| S3 | Programming languages and tools | Setting up the development environment and dependencies |
| B6 | Commitment to keeping up to date | Engaging with current ML resources and research |
