# ROI Analysis for ML Projects

> Translating model performance into business value is critical for securing stakeholder buy-in and demonstrating impact.

## The Framework

ROI quantifies the financial return of deploying a model:

$$\text{ROI} = \frac{\text{Gain from Model} - \text{Cost of Model}}{\text{Cost of Model}} \times 100\%$$

## Mapping Metrics to Business Value

| ML Metric | Business Translation |
|-----------|---------------------|
| Precision ↑ | Fewer false alarms → less wasted investigation time |
| Recall ↑ | Fewer missed positives → less revenue lost to undetected fraud |
| MAE ↓ | More accurate demand forecasts → less excess inventory |
| Accuracy ↑ | Fewer errors overall → higher customer satisfaction |

## Worked Example: Churn Prediction

```python
import numpy as np

# Business parameters
n_customers = 10000
churn_rate = 0.10                   # 10% churn
avg_customer_value = 500            # £500/year per customer
retention_cost = 50                 # £50 per retention intervention
retention_success_rate = 0.40       # 40% of interventions succeed

# Without model: no intervention
lost_revenue_no_model = n_customers * churn_rate * avg_customer_value
print(f"Revenue lost without model: £{lost_revenue_no_model:,.0f}")

# With model (recall=0.80, precision=0.60)
recall = 0.80
precision = 0.60
true_churners = n_customers * churn_rate         # 1,000
predicted_churners = true_churners * recall / precision  # ~1,333

# Cost of interventions
intervention_cost = predicted_churners * retention_cost
# Revenue saved
saved = true_churners * recall * retention_success_rate * avg_customer_value

roi = ((saved - intervention_cost) / intervention_cost) * 100

print(f"Customers targeted: {predicted_churners:,.0f}")
print(f"Intervention cost: £{intervention_cost:,.0f}")
print(f"Revenue saved: £{saved:,.0f}")
print(f"ROI: {roi:.1f}%")
```

## Communicating to Stakeholders

When presenting to non-technical audiences:

1. **Lead with the business metric** ("This model saves £120,000/year"), not the ML metric.
2. **Show the counterfactual** — what happens without the model.
3. **Use simple visuals** — bar charts comparing "with model" vs "without model."
4. **Acknowledge limitations** — explain the precision/recall tradeoff in plain English.

!!! tip "Workplace Tip"
    Frame every ML project in terms of cost savings, revenue gained, or time saved. Stakeholders rarely care about F1 scores — they care about business outcomes.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K5 | Machine Learning workflows | Translating model performance into business impact |
