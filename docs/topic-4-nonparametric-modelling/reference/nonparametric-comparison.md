# Non-Parametric Comparison

> A quick reference guide comparing the strengths and weaknesses of different non-parametric models.

## Comparison Table

| Model | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **k-NN** | Simple, no training phase | Slow inference, sensitive to scale | Baselines, simple spatial data |
| **SVM (Kernel)** | Powerful on complex boundaries | Slow to train on large data | Medium-sized complex datasets |
| **Decision Tree** | Highly interpretable | Prone to overfitting | Baselines, rule extraction |
| **Random Forest** | Robust, prevents overfitting | Can be slow to predict | General purpose tabular data |
| **XGBoost** | Extremely accurate, fast | Prone to overfitting if tuned poorly | Winning tabular competitions |

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.2 | Advanced ML techniques | Tree-based models, ensemble methods, KNN, SVM |
| K4.4 | Trade-offs in selecting algorithms | Comparing parametric vs non-parametric approaches |
| S4 | ML and optimisation | Hyperparameter tuning, ensemble construction, model selection |
| B1 | Curiosity and creativity | Exploring when non-parametric methods outperform parametric ones |
| B5 | Integrity in presenting conclusions | Avoiding overfitting; honest reporting of generalisation performance |
