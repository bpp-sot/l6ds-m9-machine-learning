# How to Execute Multi-Class Classification

> Standard models like `LogisticRegression` natively output binary Yes/No decisions. When predicting precisely between 3 or more categories (e.g. `Low`, `Medium`, `High`), specialized mathematics are required.

## Method 1: One-vs-Rest (OvR)

The most common logical strategy is **One-vs-Rest (OvR)**. 

If predicting between `Apple`, `Banana`, and `Orange`, the algorithm natively constructs **3 entirely separate binary models**:
1. *Model 1:* `Apple` vs. `Not Apple` (Probability: 12%)
2. *Model 2:* `Banana` vs. `Not Banana` (Probability: 81%)
3. *Model 3:* `Orange` vs. `Not Orange` (Probability: 7%)

The system reviews all probabilities simultaneously and strictly selects the highest mathematical value (Banana).

```python
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# The 'penguins' dataset contains 3 unique species natively
df = sns.load_dataset('penguins').dropna()

# Extract numeric features and categorical target
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate with 'multi_class=ovr' explicitly
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
```

??? example "Expected Output"
    ```text
                  precision    recall  f1-score   support

          Adelie       1.00      0.97      0.99        38
       Chinstrap       0.95      1.00      0.98        21
          Gentoo       1.00      1.00      1.00        25

        accuracy                           0.99        84
    ```

## Method 2: Multinomial Native Support

Some algorithms (Decision Trees, Random Forests, Naive Bayes) inherently support multi-class boundaries topologically without explicitly requiring forced OvR binary loops. 

They mathematically calculate Gini impurities dynamically across all 3 classes simultaneously inside every branch.

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest natively maps Multiple Classes
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# Output predictions
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.3f}")
```

!!! tip "Workplace Tip"
    When utilizing complex multi-class data natively, strictly monitor the `Macro F1-Score` over standard Accuracy. Accuracy inherently masks explicit failure on minority classes mathematically.

## KSB Mapping

| KSB | Description | How This Addresses It |
|-----|-------------|-------------------------------|
| K4.1 | Statistical models and methods | Understanding the statistical basis of regression and classification |
| K4.2 | ML and AI techniques | Implementing and comparing supervised learning algorithms |
| K4.4 | Resource constraints and trade-offs | Model complexity vs interpretability; computational cost |
| S1 | Scientific methods and hypothesis testing | Formulating hypotheses and testing with rigorous validation |
| S4 | Building models and validating | Cross-validation, train/test evaluation, performance metrics |
| B5 | Impartial, hypothesis-driven approach | Honest evaluation of model performance and limitations |
