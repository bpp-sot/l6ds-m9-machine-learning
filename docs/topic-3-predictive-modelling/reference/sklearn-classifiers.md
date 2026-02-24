# Scikit-Learn Classifiers Reference

> Quick lookup for commonly utilized categorical machine learning algorithms within `sklearn`.

## Linear Models

### `LogisticRegression`
**Use Case:** Baseline classification, binary probabilities.
**Key Parameters:**
* `C`: Controls regularisation strength (Inverse - smaller means stronger).
* `penalty`: Defines regularisation type (`'l1'`, `'l2'`).
* `class_weight='balanced'`: Automatically adjust weights for imbalanced data.

## Tree-Based Models

### `DecisionTreeClassifier`
**Use Case:** Fast, explainable non-linear rules.
**Key Parameters:**
* `max_depth`: Limits tree depth to strictly prevent overfitting.
* `min_samples_split`: Minimum rows required before making a split.

### `RandomForestClassifier`
**Use Case:** Robust, general-purpose ensemble model.
**Key Parameters:**
* `n_estimators`: Count of trees to randomly construct.
* `max_depth`: Controls complexity of individual trees.

### `GradientBoostingClassifier`
**Use Case:** High-performance, sequential error-correcting model.
**Key Parameters:**
* `learning_rate`: Step size for each tree's correction.
* `n_estimators`: Total sequential stages.

## Advanced Models

### `SVC` (Support Vector Classifier)
**Use Case:** Complex geometric boundary separation.
**Key Parameters:**
* `kernel`: Feature space transformation (`'linear'`, `'rbf'`, `'poly'`).
* `C`: Margin hardness scale.

### `MLPClassifier` (Neural Network)
**Use Case:** Deep learning approximation.
**Key Parameters:**
* `hidden_layer_sizes`: Structure of the network (e.g., `(100, 50)`).
* `activation`: Logic function (`'relu'`, `'logistic'`).

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
