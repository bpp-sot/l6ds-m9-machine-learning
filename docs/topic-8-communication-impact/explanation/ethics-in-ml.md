# Ethics in Machine Learning

> A model is only as ethical as the data it was trained on and the people who built it.

## The Problem of Bias
Models do not have morals. They pattern-match history. If historical data contains human biases (e.g., denying more loans to minority neighborhoods), the model will learn that rule and scale it instantly, codifying the racism mathematically under a veneer of "objective algorithmic accuracy."

## Real-World Examples
*   **Hiring Algorithms:** An AI trained on 10 years of successful resumes for software engineers might learn that being male is a strong predictor of success, simply because historical hiring was skewed male. It would then actively penalize female applicants.
*   **Healthcare Algorithms:** An algorithm predicting healthcare needs learned to allocate fewer resources to Black patients than White patients with similar health profiles, because it used historical *spending* as a proxy for *need*, and Black patients historically had less money spent on them due to systemic inequality.

## Your Responsibility
As a data scientist, you are the last line of defence. It is your job to slice your validation metrics by demographic groups to ensure the model performs equally well (or equally badly) for all protected classes.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| S5 | Deployment, value assessment, and ROI | Translating model performance into business impact |
| S6 | Communicate through storytelling and visualisation | Presenting ML results to non-technical stakeholders |
| B4 | Consideration of organisational goals | Framing technical results in terms of business objectives |
| B1 | Inquisitive approach | Exploring creative ways to explain model behaviour |
