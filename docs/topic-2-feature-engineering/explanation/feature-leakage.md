# Explanation: Information Leakage via Features

## Conceptual Overview
While we previously discussed Data Leakage manifesting via procedural errors (like using `StandardScaler` prior to splitting), **Feature Leakage** is a much more insidious logical error where the engineered feature itself contains the answer key to the prediction.

Feature leakage mathematically guarantees 100% precision during training and 0% utility in production.

## Case Studies in Sabotage

### Example 1: The "Account Closed Date"
**The Goal:** Build a model predicting if an active bank customer will Churn (close their account) in the next 30 days.

**The Leak:** You include a feature called `Days_Since_Account_Closed`. If the account is open, this value is `NaN`. If closed, it is an integer. 

**The Fallacy:** The model realizes that every time `Days_Since_Account_Closed` is an integer, the target `Churn` is exactly `1`. It builds an entire Random Forest ignoring everything else based on this one column. In production, an active customer will *always* have `NaN` for this field, rendering the model useless.

### Example 2: The E-Commerce "Cart Total"
**The Goal:** Predict if a user browsing a website will complete a purchase today.

**The Leak:** You include the feature `Checkout_Cart_Total_Value`. 

**The Fallacy:** `Checkout_Cart_Total_Value` is only populated *after* the user decides to buy something. You are using an event that happens at $T_{+1}$ to predict the classification occurring at $T_{0}$. In production, when the user is browsing the homepage, the cart value is $0$.

## Identifying Feature Leakage

If your cross-validated model suddenly achieves 99.8% precision on an actively difficult real-world problem, you have almost certainly leaked the target via feature engineering.

**Validation Checklist:**
1. Does this feature physically exist in the universe *at the exact moment* the prediction must be made?
2. Is this feature a direct downstream proxy of the target variable?
3. Did I extract this feature from a database column that is updated retrospectively?

## Connection to Practice
In the M9 summative project, examiners rigorously inspect your final dataset columns to ensure no temporal proxies slipped into your matrix. Ensuring strict chronological discipline across your `X` feature matrix is paramount.
