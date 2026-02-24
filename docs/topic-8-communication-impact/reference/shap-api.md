# SHAP Library Reference

> SHAP connects game theory with machine learning to provide robust, globally consistent explanations.

## Quick API

*   **TreeExplainer:** Optimised specifically for Tree-based models (XGBoost, Random Forest). Extremely fast.
    ```python
    explainer = shap.TreeExplainer(model)
    ```
*   **LinearExplainer:** For linear models (Logistic Regression, Linear Regression).
    ```python
    explainer = shap.LinearExplainer(model, X_train)
    ```
*   **KernelExplainer:** The fallback for *any* model (including neural networks or complex ensembles). Very slow.
    ```python
    explainer = shap.KernelExplainer(model.predict, X_train_summary)
    ```

## Standard Plots

*   `shap.plots.waterfall(shap_values[0])`: Explains a single prediction.
*   `shap.plots.force(shap_values[0])`: Alternative horizontal view for a single prediction.
*   `shap.plots.beeswarm(shap_values)`: Global view showing feature intensity and impact across the whole dataset.
*   `shap.plots.bar(shap_values)`: Standard global feature importance.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
