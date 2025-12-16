"""
Linear Regression (NumPy-only).

This module implements an sklearn-like LinearRegression estimator:

- Ordinary Least Squares (OLS)
- Ridge regression (L2 regularization)
- Two solvers:
    - "normal": closed-form using solve/lstsq (stable for small/medium data)
    - "gd": batch gradient descent (useful when you don't want matrix solves)

Supports:
- fit_intercept
- single-output and multi-output regression
- score() returning R^2 (with sklearn-style constant-y behavior)

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>> X = np.array([[1., 0.],
...               [0., 1.],
...               [1., 1.],
...               [2., 1.]])
>>> y = np.array([3., 4., 6., 8.])  # y = 2*x0 + 3*x1 + 1
>>> model = LinearRegression(fit_intercept=True, solver="normal")
>>> _ = model.fit(X, y)
>>> np.round(model.coef_, 6).tolist()
[2.0, 3.0]
>>> float(np.round(model.intercept_, 6))
1.0
>>> np.round(model.predict([[3., 2.]]), 6).tolist()
[13.0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

__all__ = ["LinearRegression"]

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


def _ensure_y_float(y, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D; got {arr.ndim}D.")
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


def _as_2d_y(y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (y2d, was_1d)."""
    if y.ndim == 1:
        return y.reshape(-1, 1), True
    return y, False


def _add_intercept_column(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=float)
    return np.concatenate([ones, X], axis=1)


@dataclass
class LinearRegression:
    """
    Linear Regression estimator (NumPy-only).

    Parameters
    ----------
    fit_intercept : bool, default=True
        If True, learns an intercept term.
    solver : {"normal", "gd"}, default="normal"
        - "normal": closed-form (solve / lstsq) with optional ridge.
        - "gd": batch gradient descent with optional ridge.
    l2 : float, default=0.0
        L2 regularization strength (ridge). Must be >= 0.
        Note: intercept is NOT regularized.
    max_iter : int, default=5000
        Max iterations for solver="gd".
    lr : float, default=0.1
        Learning rate for solver="gd".
    tol : float, default=1e-10
        Convergence tolerance for solver="gd" (on gradient norm OR relative loss change).
    eps : float, default=1e-12
        Numerical floor used to avoid division issues.

    Attributes
    ----------
    coef_ : ndarray
        - shape (n_features,) for single-output
        - shape (n_outputs, n_features) for multi-output
    intercept_ : float or ndarray
        - float for single-output
        - shape (n_outputs,) for multi-output
    n_features_in_ : int
        Number of input features seen during fit.
    n_outputs_ : int
        Number of outputs.
    n_iter_ : int
        Number of iterations (1 for "normal"; <= max_iter for "gd").
    loss_history_ : ndarray or None
        For solver="gd", stores per-iteration objective values (useful for plotting).
    """

    fit_intercept: bool = True
    solver: Literal["normal", "gd"] = "normal"
    l2: float = 0.0
    max_iter: int = 5000
    lr: float = 0.1
    tol: float = 1e-10
    eps: float = 1e-12

    # learned
    coef_: Optional[np.ndarray] = None
    intercept_: Optional[np.ndarray] = None
    n_features_in_: Optional[int] = None
    n_outputs_: Optional[int] = None
    n_iter_: int = 0
    loss_history_: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.solver not in ("normal", "gd"):
            raise ValueError("solver must be 'normal' or 'gd'.")
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("fit_intercept must be bool.")
        if not (isinstance(self.l2, (int, float)) and np.isfinite(self.l2) and self.l2 >= 0):
            raise ValueError("l2 must be a finite non-negative number.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not (isinstance(self.lr, (int, float)) and np.isfinite(self.lr) and self.lr > 0):
            raise ValueError("lr must be a finite positive number.")
        if not (isinstance(self.tol, (int, float)) and np.isfinite(self.tol) and self.tol >= 0):
            raise ValueError("tol must be a finite non-negative number.")
        if not (isinstance(self.eps, (int, float)) and np.isfinite(self.eps) and self.eps > 0):
            raise ValueError("eps must be a finite positive number.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_y_float(y, "y")

        n_samples, n_features = X_arr.shape
        y2d, was_1d = _as_2d_y(y_arr)

        if y2d.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples.")

        self.n_features_in_ = int(n_features)
        self.n_outputs_ = int(y2d.shape[1])

        # Build design matrix
        Xd = _add_intercept_column(X_arr) if self.fit_intercept else X_arr
        p = Xd.shape[1]  # number of parameters (incl intercept if used)

        if self.solver == "normal":
            w = self._fit_normal(Xd, y2d, p)
            self.n_iter_ = 1
            self.loss_history_ = None
        else:
            w = self._fit_gd(Xd, y2d, p)
            # n_iter_ and loss_history_ are set in _fit_gd

        # unpack parameters -> coef_ / intercept_
        if self.fit_intercept:
            intercept = w[0]          # (n_outputs,)
            coef = w[1:].T            # (n_outputs, n_features)
        else:
            intercept = np.zeros((self.n_outputs_,), dtype=float)
            coef = w.T                # (n_outputs, n_features)

        # match sklearn-like shapes for single-output
        if was_1d:
            self.intercept_ = float(intercept[0])
            self.coef_ = coef[0].copy()
        else:
            self.intercept_ = intercept.copy()
            self.coef_ = coef.copy()

        return self

    def _fit_normal(self, Xd: np.ndarray, y2d: np.ndarray, p: int) -> np.ndarray:
        # Solve (Xd^T Xd + l2*I) w = Xd^T y
        XtX = Xd.T @ Xd
        Xty = Xd.T @ y2d

        if self.l2 > 0:
            reg = np.eye(p, dtype=float) * float(self.l2)
            if self.fit_intercept:
                reg[0, 0] = 0.0  # do not regularize intercept
            A = XtX + reg
        else:
            A = XtX

        try:
            w = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            # fall back to least squares if singular/ill-conditioned
            w, *_ = np.linalg.lstsq(A, Xty, rcond=None)

        # w shape: (p, n_outputs)
        return w.astype(float, copy=False)

    def _objective(self, Xd: np.ndarray, y2d: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
        n = Xd.shape[0]
        resid = Xd @ w - y2d
        mse = 0.5 * float(np.sum(resid * resid)) / max(n, 1)
        reg = 0.5 * float(self.l2) * float(np.sum((mask * w) ** 2))
        return mse + reg

    def _fit_gd(self, Xd: np.ndarray, y2d: np.ndarray, p: int) -> np.ndarray:
        n = Xd.shape[0]
        w = np.zeros((p, y2d.shape[1]), dtype=float)

        # mask: 0 for intercept row if fit_intercept else all ones
        mask = np.ones_like(w, dtype=float)
        if self.fit_intercept:
            mask[0, :] = 0.0

        loss_hist = []
        prev_loss = None

        for it in range(1, int(self.max_iter) + 1):
            # gradient of (1/2n)||Xw-y||^2 + (l2/2)||mask*w||^2
            resid = Xd @ w - y2d
            grad = (Xd.T @ resid) / max(n, 1)
            if self.l2 > 0:
                grad = grad + float(self.l2) * (mask * w)

            grad_norm = float(np.linalg.norm(grad))
            w = w - float(self.lr) * grad

            loss = self._objective(Xd, y2d, w, mask)
            loss_hist.append(loss)

            # stop by gradient norm
            if grad_norm <= float(self.tol):
                self.n_iter_ = it
                self.loss_history_ = np.asarray(loss_hist, dtype=float)
                return w

            # stop by relative loss change
            if prev_loss is not None:
                denom = max(abs(prev_loss), float(self.eps))
                rel_change = abs(prev_loss - loss) / denom
                if rel_change <= float(self.tol):
                    self.n_iter_ = it
                    self.loss_history_ = np.asarray(loss_hist, dtype=float)
                    return w
            prev_loss = loss

        self.n_iter_ = int(self.max_iter)
        self.loss_history_ = np.asarray(loss_hist, dtype=float)
        return w

    def _check_is_fitted(self) -> None:
        if self.coef_ is None or self.intercept_ is None or self.n_features_in_ is None:
            raise RuntimeError("LinearRegression is not fitted. Call fit(X, y) first.")

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")

        if isinstance(self.intercept_, float):
            y_pred = X_arr @ self.coef_ + float(self.intercept_)
            return y_pred.astype(float, copy=False)

        # multi-output
        # coef_: (n_outputs, n_features)
        y_pred = X_arr @ self.coef_.T + self.intercept_
        return y_pred.astype(float, copy=False)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R^2 score.

        Notes (sklearn-like):
        - If y is constant:
            - returns 1.0 if predictions are perfect
            - otherwise returns 0.0
        """
        y_true = _ensure_y_float(y, "y")
        y_pred = self.predict(X)

        y2d, _ = _as_2d_y(y_true)
        yp2d, _ = _as_2d_y(np.asarray(y_pred))

        if y2d.shape != yp2d.shape:
            raise ValueError("Predictions and y have incompatible shapes.")

        # per-output R^2 then average
        r2s = []
        for j in range(y2d.shape[1]):
            yt = y2d[:, j]
            yp = yp2d[:, j]
            ss_res = float(np.sum((yt - yp) ** 2))
            y_mean = float(np.mean(yt))
            ss_tot = float(np.sum((yt - y_mean) ** 2))

            if ss_tot == 0.0:
                r2s.append(1.0 if ss_res == 0.0 else 0.0)
            else:
                r2s.append(1.0 - ss_res / ss_tot)

        return float(np.mean(r2s))
