# Neural Networks (Perceptrons)

> A Neural Network mathematically simulates the biological architecture of a human brain, allowing infinite complexity but requiring massive volumes of data to converge.

## What You Will Learn
- Define the Multi-Layer Perceptron (MLP) architecture
- Train an `MLPClassifier` utilizing Scikit-Learn
- Distinguish between Deep Learning and standard Machine Learning

## Prerequisites
- Completed Topic 1 (Data Preparation)
- Understanding of Matrix mathematics and Activation functions

## Step 1: Network Architecture

Unlike Linear Regression (which computes one global equation), a Neural Network structures thousands of tiny individual mathematical equations called **Neurons** (or Perceptrons) inside "Hidden Layers".

1. **Input Layer:** Receives the raw dataset columns natively.
2. **Hidden Layers:** Each internal neuron applies a linear formula ($W \cdot X + B$) and passes the numerical result through a non-linear activation (like `ReLU`).
3. **Output Layer:** Condenses the final hidden logic into a prediction natively.

When an architecture contains 3 or more Hidden Layers computationally, it transitions from basic Machine Learning strictly into the domain of "Deep Learning".

## Step 2: Implementation

While TensorFlow and PyTorch dominate the Deep Learning industry, Scikit-Learn provides a robust baseline `MLPClassifier` designed specifically to train basic neural topology identically.

We will explicitly train a Perceptron to map synthetically generated circle clusters natively.

```python
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# 1. Synthesize highly non-linear target mapping
X, y = make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Instantiate and define the Hidden Topology
# hidden_layer_sizes=(10, 10) equates to 2 Hidden Layers, 10 neurons each.
# max_iter is required to mathematically permit the Gradient Descent time to converge natively.
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 3. Generate predictions
preds = mlp.predict(X_test)

print(f"Perceptron Array Accuracy: {accuracy_score(y_test, preds):.2f}")
```

??? example "Expected Output"
    ```text
    Perceptron Array Accuracy: 0.98
    ```

## Step 3: Standardisation is Strictly Required

Similar to Support Vector Machines, Neural Networks internally rely on physical Gradient Descent weight calculation routines.

If `Income` ranges up to strictly `100,000` and `Age` tops out at `90`, the network mathematics explicitly will bias massively uniquely toward the `Income` variable natively.

**You must forcefully execute `StandardScaler` strictly on your feature matrix prior to initializing any `MLPClassifier`.**

!!! info "Assessment Connection"
    For your EPA, explicitly deploying a Neural Network blindly on a 300-row tabular dataset is considered an analytical error. Neural Networks natively require thousands or millions of observations mathematically to prevent catastrophic overfitting intelligently. Document exactly why you actively chose an Ensemble instead natively to secure top marks.

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| S13 | Apply ML algorithms | Deploying explicit Multi-Layer Perceptrons |
| K5 | Machine Learning workflows | Selecting algorithm complexity corresponding strictly to explicit scale |
