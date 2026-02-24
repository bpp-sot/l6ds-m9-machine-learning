# Machine Learning in the Workplace

> The most difficult part of deploying machine learning is rarely the code; it is the culture change.

## The Integration Challenge

Building an accurate model is only 20% of the battle. The remaining 80% is getting people to actually use it.

1.  **Trust:** If end-users (e.g., call centre reps, warehouse managers) don't trust the model, they will ignore its recommendations. Trust is built through transparency and involving them in the design process.
2.  **Workflow Disruption:** A model that requires a user to open a new tab, copy-paste an ID, and wait 30 seconds for a prediction will fail. Models must be integrated silently and seamlessly into existing workflows (e.g., Salesforce, the ERP system).
3.  **Feedback Loops:** Every prediction the model makes needs a real-world outcome attached to it eventually to know if it was right. If a user overrides a model recommendation, you need to capture *why* they overrode it to retrain the model later.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
