# How to Save and Load Machine Learning Models

> Training a Random Forest on a large dataset can take hours. You must save the trained model to disk so you do not have to retrain it every time you need predictions.

## Using `joblib`

The standard `pickle` module works, but scikit-learn recommends `joblib` because it is optimised for large NumPy arrays (which underpin all model weights).

### Saving a Trained Model

```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save to disk
joblib.dump(model, "random_forest_v1.pkl")
print("Model saved successfully.")
```

### Loading for Inference

```python
import joblib

# Load the saved model
loaded_model = joblib.load("random_forest_v1.pkl")

# Use it for predictions without retraining
preds = loaded_model.predict(X_test)
print(f"Accuracy: {loaded_model.score(X_test, y_test):.2f}")
```

## Versioning Best Practices

1. **Include a version number** in the filename (e.g., `model_v2.pkl`) so you can roll back.
2. **Save alongside metadata** — record the training date, dataset hash, hyperparameters, and evaluation scores.
3. **Never load untrusted `.pkl` files** — pickle can execute arbitrary code during deserialisation.

!!! warning "Common Pitfall"
    The scikit-learn version used to save the model must match the version used to load it. Upgrading scikit-learn can break deserialisation. Pin your version in `requirements.txt`.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
