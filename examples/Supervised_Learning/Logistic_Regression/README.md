# Logistic Regression

This directory contains example code and notes for **Logistic Regression** in supervised learning.

## Algorithm

Logistic Regression is a linear classifier that models class probabilities.

### Binary case (sigmoid)

Given features \(x \in \mathbb{R}^d\), the model predicts:

\[
p(y=1 \mid x) = \sigma(w^\top x + b),
\quad
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

We fit \((w, b)\) by minimizing the **average log loss**:

\[
\mathcal{L}(w,b)
= -\frac{1}{n}\sum_{i=1}^n
\Big(
y_i \log p_i + (1-y_i)\log(1-p_i)
\Big)
+ \frac{\lambda}{2}\lVert w\rVert_2^2
\]

- The optional \(L2\) term (with strength \(\lambda\)) discourages large weights and can improve generalization.
- The intercept \(b\) is typically **not** regularized.

### Multiclass case (softmax / multinomial)

For \(K\) classes, we use softmax:

\[
p(y=k\mid x) =
\frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)}
\]

and minimize cross-entropy (plus optional L2 regularization on \(W\)).

## Key hyperparameters

- `learning_rate`: gradient descent step size
- `max_iter`: maximum number of iterations
- `tol`: stopping tolerance (loss improvement threshold)
- `l2`: L2 regularization strength (0 = no regularization)
- `batch_size`: `None` for full-batch GD; an integer enables mini-batch training
- `multi_class`: `"auto"` or `"multinomial"`

## Data

Logistic regression expects:

- `X`: feature matrix of shape `(n_samples, n_features)`
- `y`: labels of shape `(n_samples,)`

Notes:
- Features are often standardized (mean 0, std 1) to improve optimization stability.
- Labels can be integers or strings; the implementation internally encodes them.

