# Linear Regression

This directory contains example code and notes for the Linear Regression algorithm
in supervised learning.

## Overview
Linear Regression is a supervised learning method for predicting a **continuous**
target value. It assumes the target can be approximated by a linear function of
the input features.

## Algorithm
We model predictions as:

    y_hat = X @ w + b

where:
- `X` is the feature matrix (n_samples, n_features)
- `w` is the weight vector (n_features,)
- `b` is the intercept (scalar)

Training finds parameters `(w, b)` that minimize **mean squared error (MSE)**:

    MSE = mean((y - y_hat)^2)

Optionally, we can add **L2 (ridge) regularization** to improve numerical
stability when features are correlated:

    objective = MSE + l2 * ||w||^2

Common implementation options / hyperparameters:
- `fit_intercept`: whether to learn an intercept term `b`
- `solver`: "normal" (closed-form) or "gd" (gradient descent)
- `l2`: L2 regularization strength (0.0 means plain OLS)
- `lr`, `max_iter`, `tol`: optimization controls for gradient descent

## Data
Linear Regression uses labeled regression data:
- Features: numeric matrix `X` with shape `(n_samples, n_features)`
- Targets: numeric vector `y` with shape `(n_samples,)`

In the example notebook, we typically:
1. Load a regression dataset (diabetes dataset from `sklearn.datasets`).
2. Split into train/test sets.
3. Standardize features using training statistics, then apply the same transform
   to the test set to avoid data leakage.
4. Fit the model on the training set and evaluate on the test set.
