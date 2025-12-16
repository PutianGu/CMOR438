# Linear Regression

This directory contains example code and notes for the **Linear Regression** algorithm
in supervised learning.

## Algorithm

**Linear Regression** models a continuous target \(y\) as a linear function of the input features \(X\):

\[
\hat{y} = Xw + b
\]

The goal is to choose parameters \(w\) (weights) and \(b\) (intercept) that minimize the **mean squared error (MSE)** on the training data:

\[
\min_{w,b}\ \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (x_i^\top w + b)\right)^2
\]

### Optional: Ridge (L2 Regularization)
To improve stability (especially with correlated features), we can add an L2 penalty on the weights:

\[
\min_{w,b}\ \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (x_i^\top w + b)\right)^2 + \lambda \lVert w \rVert_2^2
\]

In our implementation:
- `l2` corresponds to \(\lambda\)
- the **intercept is not regularized**

### Key Hyperparameters (in `rice_ml`)
- `fit_intercept` (bool): whether to learn an intercept term \(b\)
- `solver`:
  - `"normal"`: closed-form solution using linear algebra
  - `"gd"`: batch gradient descent (iterative optimization)
- `l2` (float): ridge regularization strength (0.0 means plain OLS)
- `lr`, `max_iter`, `tol`: only used for `"gd"` to control optimization

## Data

Linear Regression is a **supervised** algorithm:
- **Inputs:** numeric feature matrix \(X\) with shape `(n_samples, n_features)`
- **Targets:** continuous target vector \(y\) with shape `(n_samples,)`  
  (or multi-output targets with shape `(n_samples, n_outputs)`)

### Dataset used in the example notebook
The example notebook uses the **Diabetes** regression dataset (from `sklearn.datasets`):
- `X`: 10 standardized baseline variables (features)
- `y`: a quantitative measure of disease progression (continuous)

### Preprocessing
Even if a dataset is already reasonably scaled, we typically **standardize** features (zero mean, unit variance), especially when using gradient descent:
- improves numerical stability
- helps gradient descent converge faster and more reliably
- makes coefficient magnitudes more comparable across features

In the notebook:
- we compute standardization parameters from the **training split only**
- apply the same transform to the test split (to avoid data leakage)

## Files
- `Linear_Regression.ipynb`: walkthrough notebook (data loading, fitting OLS/Ridge/GD, evaluation, plots)
- `README.md`: this guide

