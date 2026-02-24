# LIME Library Reference

> LIME explains the predictions of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.

## Quick API (Tabular Data)

The core object is the `LimeTabularExplainer`. You must initialize it with the training data.

```python
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np_array_of_training_data,
    feature_names=['age', 'income', 'credit_score'],
    class_names=['Deny', 'Approve'],
    mode='classification' # or 'regression'
)
```

## Extracting an Explanation

You explain a single instance (row) by passing the data and the model's prediction function.

```python
exp = explainer.explain_instance(
    data_row=single_np_array_row, 
    predict_fn=model.predict_proba, # Must output probabilities for classification!
    num_features=5 # How many features to show in the explanation
)

# Render in a Jupyter notebook
exp.show_in_notebook(show_table=True)

# Or print as a list
print(exp.as_list())
```

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
