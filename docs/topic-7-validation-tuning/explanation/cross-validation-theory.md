# Why Cross-Validation Works

> A single train-test split is subject to luck. Cross-validation eliminates this.

## The Theory
If you split your dataset 80/20 just once, there is a chance that the 20% in the test set contains all the easy examples, or all the hard outliers. Your evaluation metric will be misleadingly high or low.

Cross-validation (specifically K-Fold) solves this by rotating the test set.
If we use 5-Folds:
1.  We split data into 5 chunks.
2.  We train on chunks 1,2,3,4 and test on chunk 5.
3.  We train on chunks 2,3,4,5 and test on chunk 1.
4.  And so on.

## The Benefit
Every single data point in your dataset gets to be in the test set exactly once. By averaging the 5 scores, you get a highly robust, "luck-free" estimate of how the model will perform in reality.

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |
