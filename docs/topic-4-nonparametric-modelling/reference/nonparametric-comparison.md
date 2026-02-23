# Reference: Nonparametric Comparison

This page contains quick-lookup information for nonparametric comparison.

## Key Methods and Parameters

| Method | Parameters | Description |
|--------|------------|-------------|
| `fit()` | `X`, `y` | Fits the model or transformer to the data |
| `transform()` | `X` | Applies the transformation |
| `predict()` | `X` | Generates predictions |

## Common Syntax

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Standard boilerplate
pipeline = make_pipeline(StandardScaler(), ...)
pipeline.fit(X_train, y_train)
```

## Comparison Metrics

When comparing approaches for nonparametric comparison, consider:

1. **Accuracy**: How well does it perform?
2. **Interpretability**: How easily can you explain it?
3. **Speed**: How fast does it run?
