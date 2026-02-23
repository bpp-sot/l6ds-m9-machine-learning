# Naive Bayes Classifiers

> "All features are created equal, and all features operate entirely independently of each other. This is statistically impossible, but mathematically brilliant."

## What You Will Learn

- Understand Bayes' Theorem for conditional probability
- Implement `MultinomialNB` for Text Classification
- Implement `GaussianNB` for Continuous tabular data

## Prerequisites

- [DateTime & Text Features](../../topic-2-feature-engineering/tutorials/datetime-text-features.md)

## Step 1: Bayes' Theorem

Naive Bayes is a probabilistic classifier based entirely on applying Bayes' theorem with a strong (naive) assumption of independence between the features. 

Bayes' theorem calculates the probability of an event ($A$) given that some prior condition ($B$) is true:

\\[
P(A|B) = \\frac{P(B|A) \\times P(A)}{P(B)}
\\]

**The "Naive" Assumption:** The algorithm assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. E.g., If calculating the probability of a fruit being an "Apple", it assumes that the color "Red", the shape "Round", and the diameter "3 inches" all contribute to the probability *independently*. 

## Step 2: Implementation (Text Data)

Naive Bayes is historically the industry-standard baseline for **Sentiment Analysis** and **Spam Filtering**. Because it only requires calculating raw probabilities (counting word occurrences), it trains almost instantaneously, even on datasets with 100,000 features.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Raw Text Data
texts = [
    "I loved this movie, it was fantastic", 
    "Terrible film, completely boring",
    "Great acting and wonderful script",
    "What a waste of time"
]
labels = [1, 0, 1, 0] # 1 = Positive, 0 = Negative

# 2. Vectorization (Word frequencies)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 3. Train Naive Bayes
# MultinomialNB is designed specifically for discrete counts (like word frequencies)
clf = MultinomialNB()
clf.fit(X, labels)

# 4. Predict novel text
new_review = ["The movie was a waste and boring"]
new_transformed = vectorizer.transform(new_review)
prediction = clf.predict(new_transformed)

print(f"Review: '{new_review[0]}'")
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

## Step 3: Gaussian Naive Bayes (Tabular Data)

If your dataset contains continuous numeric columns (like `Salary` or `Age`), you cannot count discrete frequencies. Instead, you use `GaussianNB`, which estimates the probabilities by assuming the features follow a normal (Gaussian) distribution.

```python
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB

# Load tabular continuous data
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and Fit
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"Gaussian NB Accuracy: {gnb.score(X_test, y_test):.4f}")
```

## Summary

Naive Bayes is the fastest classification algorithm in existence. While Deep Neural Networks (like LSTMs or Transformers) have superseded it for highly complex Natural Language Processing, Naive Bayes remains the mandatory baseline metric. If your 100-layer Neural Network evaluating spam emails cannot beat the Naive Bayes probability matrix, your network is an over-engineered failure.

## Next Steps

→ [Model Interpretability (SHAP & LIME)](model-interpretability.md)

## KSB Mapping

| KSB | Description | How This Tutorial Addresses It |
|-----|-------------|-------------------------------|
| K1 | Mathematical Principles | Solves classification using conditional probability vectors |
| S7 | Analyse relationships | Maps unstructured text into categorical insight vectors |
