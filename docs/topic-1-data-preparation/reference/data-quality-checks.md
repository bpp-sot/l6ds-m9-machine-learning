# Data Quality Checks Reference

> A checklist for determining when your dataset is legally "clean enough" to ingest into training algorithms.

## Required Pipeline Conditions

Machine Learning algorithms computationally explicitly require the following criteria to operate without crashing or emitting fatal runtime errors:

1. **Zero Nulls/NaNs/Missings**: Algorithms strictly process continuous numbers. You must either structurally drop or rigorously impute every single missing value.
2. **Homogenous Numeric Arrays**: Algorithms do not execute on strings. Every single categorical variable must be One-Hot, Ordinal, or Frequency Encoded into numeric architectures.
3. **No Duplicate Dimensions**: Features tracking identical behaviour implicitly explicitly corrupt distance-based boundaries.
4. **Rescaled Magnitude Constraints**: Unscaled `Millions` vs `Decimals` computationally destroys K-Means and pure Neural Net initialisation bounds.

## Recommended Validation Script

Run this diagnostic function explicitly structurally across your Final DataFrame object before passing it towards `.fit()`.

```python
import pandas as pd
import numpy as np

def validate_pipeline_integrity(df):
    status = True
    print("--- 🔴 EXECUTING PIPELINE INTEGRITY AUDIT ---")
    
    # 1. Null Check
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"❌ FAILED: Detected {nulls} missing values.")
        status = False
    else:
        print("✅ PASSED: Zero missing values.")
        
    # 2. String/Object Check
    text_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(text_cols) > 0:
        print(f"❌ FAILED: Found {len(text_cols)} Text/Object columns that require Encoding: {list(text_cols)}")
        status = False
    else:
        print("✅ PASSED: All columns strictly float/int.")

    # 3. Variance / Constant Value Check
    zero_var = [col for col in df.columns if df[col].nunique() <= 1]
    if len(zero_var) > 0:
        print(f"❌ FAILED: Found strictly constant columns (Variance=0) providing zero predictive signal: {zero_var}")
        status = False
    else:
        print("✅ PASSED: All variables possess valid variance.")

    print("--- \nAUDIT COMPLETE: ", "PROCEED TO ALGORITHM 🟢" if status else "FIX AUDIT FAILURES 🔴")
    return status

# Example Usage
import seaborn as sns
df = sns.load_dataset('diamonds')
validate_pipeline_integrity(df)
```

??? example "Expected Output"
    ```text
    --- 🔴 EXECUTING PIPELINE INTEGRITY AUDIT ---
    ✅ PASSED: Zero missing values.
    ❌ FAILED: Found 3 Text/Object columns that require Encoding: ['cut', 'color', 'clarity']
    ✅ PASSED: All variables possess valid variance.
    --- 
    AUDIT COMPLETE:  FIX AUDIT FAILURES 🔴
    ```

!!! tip "Workplace Tip"
    Packaging functions strictly like `validate_pipeline_integrity()` centrally into standard utility toolsets (`utils.py`) dynamically guarantees data architects won't silently execute fatally illegal algorithms throughout live continuous production environments!
