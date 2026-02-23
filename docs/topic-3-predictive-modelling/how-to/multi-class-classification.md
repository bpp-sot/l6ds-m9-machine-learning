# How-to: Multi Class Classification

## The Problem
In your workplace projects, you will frequently encounter the need to multi class classification. This guide provides a direct solution.

## The Solution
Use the following approach:

```python
import pandas as pd
import numpy as np

def resolve_multi_class_classification(data):
    # Apply transformation
    result = data.copy()
    # Your business logic here
    return result

# Example usage:
# df_clean = resolve_multi_class_classification(df_raw)
```

## Discussion
### When to use this approach?
Use this when your dataset explicitly requires multi class classification. It is particularly useful for messy organizational data.

### Caveats
- Computationally expensive for large datasets.
- Ensure you have handled missing values prior to this step.
