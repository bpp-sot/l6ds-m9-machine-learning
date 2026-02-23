# How-to: Save and Load Trained Models

## The Problem
You trained an XGBoost model. It took 14 hours utilizing 64 cores. When you close your Jupyter Notebook or Python IDE, the RAM clears and the model is destroyed. You must save the structural weights to disk so a separate application (like a web server or Streamlit app) can load it instantly.

## The Solution
We serialize the trained Python object into a byte-stream file, commonly referred to as a `.pkl` (Pickle) or `.joblib` file.

### Using Joblib (Industry Standard for Scikit-Learn)

`Joblib` is heavily optimized for arrays and numerical data, making it vastly superior to Python's built-in `pickle` library when serializing massive Numpy/Scikit-Learn structures.

```python
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 1. Train your model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 2. Serialize to disk (The "Save")
# This creates a physical file in your directory
filename = 'rf_iris_model_v1.joblib'
joblib.dump(model, filename)
print(f"Model successfully saved to {filename}")

# --- (Imagine this is a completely different Python script, days later) ---

# 3. Deserialize from disk (The "Load")
loaded_model = joblib.load(filename)

# 4. Predict natively
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(new_data)
print(f"Prediction from loaded model: Class {prediction[0]}")
```

## Discussion

### Saving Entire Pipelines
A model is useless without its exact preceding transformations. If you only save the `RandomForestClassifier` but forget to save the `StandardScaler` or `OneHotEncoder`, you will be unable to process live data.

**Always save the entire `Pipeline` object, not just the estimator.**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

full_system = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

full_system.fit(X, y)

# Saving the Pipeline saves the scaling logic AND the model
joblib.dump(full_system, 'full_production_system.joblib')
```

### Security Caveat
> [!CAUTION]
> Pickle/Joblib files are executable byte-code. Never, under any circumstances, load a `.joblib` or `.pkl` file sent to you by an untrusted source off the internet. It can contain malicious system-level commands that will execute the moment you call `joblib.load()`.
