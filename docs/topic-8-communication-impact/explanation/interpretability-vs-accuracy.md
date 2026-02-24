# Interpretability vs Accuracy

> There is fundamentally a trade-off between how well a model performs and how easy it is to understand why.

## The Spectrum

*   **High Interpretability, Lower Accuracy:** Linear Regression, Logistic Regression, Decision Trees. If a bank denies a loan using Logistic Regression, they can point to the exact coefficient for "late payments" and say exactly why.
*   **Medium Interpretability, Medium Accuracy:** Random Forests, XGBoost. You can get global feature importance, but tracing a single prediction back to its roots is harder without external tools like SHAP.
*   **Low Interpretability, High Accuracy:** Deep Neural Networks. They achieve state-of-the-art results on image and text, but operates as a "black box" where even the creators cannot easily explain *why* it made a specific classification.

## Business Context Dictates Choice
If you are predicting whether a user will click an ad, use the most accurate black box you have. Nobody is going to sue you for showing them the wrong shoe advert. 

If you are diagnosing cancer, or denying mortgages, the law or regulators often *require* interpretability. You might have to sacrifice 2% accuracy to use a model that provides a clear "why".

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S5 | Deployment, value assessment, and ROI | Translating model performance into business impact |
| S6 | Communicate through storytelling and visualisation | Presenting ML results to non-technical stakeholders |
| B4 | Consideration of organisational goals | Framing technical results in terms of business objectives |
| B1 | Inquisitive approach | Exploring creative ways to explain model behaviour |
