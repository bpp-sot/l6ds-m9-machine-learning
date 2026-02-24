# Create a Model Card

> A model card is like a nutrition label for your machine learning model. It provides transparency.

## What goes in a Model Card?

1. **Model Details:** Algorithm type (e.g., Random Forest), date created, developer, version.
2. **Intended Use:** What is this built for? (e.g., "Predicting default risk on unsecured personal loans.")
3. **Out of Scope:** What should it NOT be used for? (e.g., "Not intended for business loans or mortgages.")
4. **Metrics:** Performance across different slices (e.g., Accuracy is 88% overall, but 85% for group A and 90% for group B). Disclosing these differences is crucial for fairness.
5. **Training Data:** A brief overview of the dataset. "Trained on 50,000 anonymised loan applications from 2018-2022."
6. **Ethical Considerations:** Known limitations or potential biases. "Historically, younger applicants have fewer data points, leading to higher false-positive rates in the 18-21 age bracket."

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
