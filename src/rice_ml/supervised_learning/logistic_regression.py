"""
Logistic Regression (NumPy-only).

This module implements a compact yet complete logistic regression estimator with:
- Binary logistic regression (sigmoid + log loss)
- Multiclass logistic regression (softmax + cross-entropy, "multinomial")
- L2 regularization (optional)
- (Mini-)batch gradient descent
- scikit-learn-like API: fit, predict, predict_proba, score

Notes
-----
- This is an educational implementation (no sklearn dependency).
- Supports non-numeric class labels (strings) by encoding internally.
- For binary problems, predict_proba returns shape (n_samples, 2).

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.logistic_regression import LogisticRegression
>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [1., 0.],
...               [1., 1.]])
>>> y = np.array([0, 0, 0, 1])
>>> clf = LogisticRegression(max_iter=2000, learning_rate=0.2, random_state=0).fit(X, y)
>>> clf.predict(X).tolist()
[0, 0, 0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union, Tuple

import numpy as np

__all__ = ["LogisticRegression"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _ensure_1d(y, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(Z: np.ndarray) -> np.ndarray:
    # stable softmax row-wise
    Z = Z.astype(float, copy=False)
    Zmax = np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z - Zmax)
    denom = np.sum(expZ, axis=1, keepdims=True)
    return expZ / np.maximum(denom, 1e-15)


def _one_hot(y_int: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((y_int.shape[0], n_classes), dtype=float)
    Y[np.arange(y_int.shape[0]), y_int] = 1.0
    return Y


@dataclass
class LogisticRegression:
    """
    Logistic Regression classifier (NumPy-only).

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for gradient descent.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tol : float, default=1e-6
        Convergence tolerance on absolute loss improvement.
    fit_intercept : bool, default=True
        Whether to learn an intercept term.
    l2 : float, default=0.0
        L2 regularization strength (>=0). Objective adds 0.5*l2*||w||^2.
        Intercept is NOT regularized.
    batch_size : int or None, default=None
        If None, use full-batch gradient descent. If int, use mini-batches.
    multi_class : {"auto", "multinomial"}, default="auto"
        - "auto": binary if 2 classes else "multinomial"
        - "multinomial": softmax regression for K>=2
    random_state : int or None, default=None
        RNG seed for shuffling mini-batches.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Class labels in sorted order (as provided in y, via np.unique).
    coef_ : ndarray
        If binary: shape (n_features,)
        If multinomial: shape (n_classes, n_features)
    intercept_ : float or ndarray
        If binary: float
        If multinomial: ndarray shape (n_classes,)
    n_iter_ : int
        Number of iterations run.
    loss_history_ : list[float]
        Loss values per iteration (averaged per sample).
    """

    learning_rate: float = 0.1
    max_iter: int = 1000
    tol: float = 1e-6
    fit_intercept: bool = True
    l2: float = 0.0
    batch_size: Optional[int] = None
    multi_class: Literal["auto", "multinomial"] = "auto"
    random_state: Optional[int] = None

    # learned
    classes_: Optional[np.ndarray] = None
    coef_: Optional[np.ndarray] = None
    intercept_: Optional[np.ndarray] = None
    n_iter_: int = 0
    loss_history_: Optional[list] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        X = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # validate params
        if not (isinstance(self.learning_rate, (int, float)) and np.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError("learning_rate must be a finite positive number.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not (isinstance(self.tol, (int, float)) and np.isfinite(self.tol) and self.tol >= 0):
            raise ValueError("tol must be a finite non-negative number.")
        if not (isinstance(self.l2, (int, float)) and np.isfinite(self.l2) and self.l2 >= 0):
            raise ValueError("l2 must be a finite non-negative number.")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
                raise ValueError("batch_size must be a positive integer or None.")
        if self.multi_class not in ("auto", "multinomial"):
            raise ValueError('multi_class must be "auto" or "multinomial".')

        # encode labels
        classes, y_int = np.unique(y_arr, return_inverse=True)
        n_classes = int(classes.size)
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification.")
        self.classes_ = classes

        mode = "multinomial" if (self.multi_class == "multinomial" or n_classes > 2) else "binary"

        rng = _rng_from_seed(self.random_state)
        n_samples, n_features = X.shape

        self.loss_history_ = []
        self.n_iter_ = 0

        if mode == "binary":
            # parameters
            w = np.zeros(n_features, dtype=float)
            b = 0.0

            # training
            last_loss = np.inf
            for it in range(1, int(self.max_iter) + 1):
                Xb, yb = self._iterate_minibatch(X, y_int, rng)

                # forward
                z = Xb @ w + (b if self.fit_intercept else 0.0)
                p = _sigmoid(z)

                # loss (avg)
                loss = self._binary_loss(p, yb, w)
                self.loss_history_.append(float(loss))

                # grads
                err = (p - yb)  # (m,)
                grad_w = (Xb.T @ err) / Xb.shape[0]
                if self.l2 > 0:
                    grad_w = grad_w + self.l2 * w
                grad_b = float(np.mean(err)) if self.fit_intercept else 0.0

                # update
                lr = float(self.learning_rate)
                w = w - lr * grad_w
                b = b - lr * grad_b

                # convergence check on full-data loss occasionally (cheap & stable)
                if it % 10 == 0:
                    full_p = _sigmoid(X @ w + (b if self.fit_intercept else 0.0))
                    full_loss = self._binary_loss(full_p, y_int, w)
                    if abs(last_loss - full_loss) <= float(self.tol):
                        self.n_iter_ = it
                        break
                    last_loss = full_loss
                self.n_iter_ = it

            self.coef_ = w
            self.intercept_ = np.array(b, dtype=float)

        else:
            # multinomial softmax regression
            W = np.zeros((n_classes, n_features), dtype=float)
            b = np.zeros(n_classes, dtype=float)
            Y_full = _one_hot(y_int, n_classes)

            last_loss = np.inf
            for it in range(1, int(self.max_iter) + 1):
                Xb, yb = self._iterate_minibatch(X, y_int, rng)
                Yb = _one_hot(yb, n_classes)

                logits = Xb @ W.T + (b if self.fit_intercept else 0.0)
                P = _softmax(logits)

                loss = self._multinomial_loss(P, Yb, W)
                self.loss_history_.append(float(loss))

                # grads
                G = (P - Yb) / Xb.shape[0]              # (m, K)
                grad_W = G.T @ Xb                       # (K, d)
                if self.l2 > 0:
                    grad_W = grad_W + self.l2 * W
                grad_b = np.mean(P - Yb, axis=0) if self.fit_intercept else 0.0

                lr = float(self.learning_rate)
                W = W - lr * grad_W
                if self.fit_intercept:
                    b = b - lr * grad_b

                if it % 10 == 0:
                    logits_full = X @ W.T + (b if self.fit_intercept else 0.0)
                    P_full = _softmax(logits_full)
                    full_loss = self._multinomial_loss(P_full, Y_full, W)
                    if abs(last_loss - full_loss) <= float(self.tol):
                        self.n_iter_ = it
                        break
                    last_loss = full_loss
                self.n_iter_ = it

            self.coef_ = W
            self.intercept_ = b

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X = _ensure_2d_float(X, "X")

        n_classes = int(self.classes_.size)
        if n_classes == 2 and self.coef_.ndim == 1:
            w = self.coef_
            b = float(self.intercept_) if self.fit_intercept else 0.0
            p1 = _sigmoid(X @ w + b)
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        else:
            W = self.coef_
            b = self.intercept_ if self.fit_intercept else 0.0
            logits = X @ W.T + b
            return _softmax(logits)

    def predict(self, X: ArrayLike) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y lengths do not match.")
        return float(np.mean(y_true == y_pred))

    # ------------------------ internal helpers ------------------------

    def _check_is_fitted(self) -> None:
        if self.classes_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LogisticRegression is not fitted. Call fit(X, y) first.")

    def _iterate_minibatch(
        self, X: np.ndarray, y_int: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.batch_size is None or self.batch_size >= X.shape[0]:
            return X, y_int

        n = X.shape[0]
        bs = int(self.batch_size)
        idx = rng.integers(0, n, size=bs)
        return X[idx], y_int[idx]

    def _binary_loss(self, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        # y is {0,1}
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * float(self.l2) * float(np.sum(w * w)) if self.l2 > 0 else 0.0
        return float(ce + reg)

    def _multinomial_loss(self, P: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
        eps = 1e-15
        P = np.clip(P, eps, 1.0)
        ce = -np.mean(np.sum(Y * np.log(P), axis=1))
        reg = 0.5 * float(self.l2) * float(np.sum(W * W)) if self.l2 > 0 else 0.0
        return float(ce + reg)
