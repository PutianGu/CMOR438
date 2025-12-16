# Logistic Regression

This directory contains example code and notes for **Logistic Regression** in supervised learning.

## Algorithm

Logistic Regression is a **linear classifier**: it computes a linear score and then converts it into a probability.

### Binary case (sigmoid)

Given a feature vector `x` (length `d`), the model predicts the probability of class 1:

- **Linear score:** `z = w·x + b`
- **Sigmoid:** `σ(z) = 1 / (1 + exp(-z))`
- **Probability:** `p(y=1 | x) = σ(w·x + b)`

We fit `(w, b)` by minimizing the **average log loss** (cross-entropy).  
Let `p_i = p(y=1 | x_i)`:

Loss(w, b) = -(1/n) * Σ [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

To discourage very large weights, we can add an L2 penalty:
Loss(w, b) = -(1/n) * Σ [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ] + (λ/2) * ||w||_2^2

Notes:
- `λ` controls regularization strength (larger `λ` → smaller weights).
- The intercept `b` is typically **not** regularized.

### Multiclass case (softmax / multinomial)

For `K` classes, we keep one weight vector per class: `w_k`, `b_k`.

- **Class score:** `s_k = w_k·x + b_k`
- **Softmax probability:**

p(y = k | x) = exp(s_k) / Σ_j exp(s_j)


The training objective is the multiclass cross-entropy loss (optionally with L2 regularization on the weight matrix).

## Data (used in the example notebook)

The example notebook uses:
- A **synthetic binary** 2D dataset (for visualization and decision boundary)
- A **synthetic multiclass** 2D dataset (for softmax regions)


## Key hyperparameters

- `learning_rate`: gradient descent step size
- `max_iter`: maximum number of iterations
- `tol`: stopping tolerance (loss improvement threshold)
- `l2`: L2 regularization strength (0 = no regularization)
- `batch_size`: `None` for full-batch GD; an integer enables mini-batch training
- `multi_class`: `"auto"` or `"multinomial"`

## Data

This example notebook uses **synthetic datasets (NumPy-only)** so it runs without external datasets
or scikit-learn. Other supervised-learning notebooks may use scikit-learn datasets; this one is intentionally self-contained for visualization. We include both **binary** and **multiclass** settings.

### Dataset A: Binary 2D Gaussian blobs (synthetic)
- **Samples**: 320 total (160 per class)
- **Features**: 2 continuous features (`x1`, `x2`)
- **Labels**: integer labels `{0,1}`
- **Generation**: two 2D Gaussian clusters with different means and similar variance
- **Purpose**: allows an intuitive 2D visualization of the learned decision boundary

### Dataset B: Multiclass 2D Gaussian blobs (synthetic)
- **Samples**: 360 total (120 per class)
- **Features**: 2 continuous features (`x1`, `x2`)
- **Labels**: string labels `{"A","B","C"}` (demonstrates non-numeric label handling)
- **Generation**: three 2D Gaussian clusters
- **Purpose**: demonstrates softmax / multinomial logistic regression and decision regions

## What to look for

- Decision boundary / decision regions in 2D
- Training loss curve (should decrease then flatten as it converges)
- Confusion matrix and accuracy on the test split
- Effect of L2 regularization on weight magnitude and boundary smoothness

Notes:
- Features are often standardized (mean 0, std 1) to improve optimization stability.
- Labels can be integers or strings; the implementation internally encodes them.

