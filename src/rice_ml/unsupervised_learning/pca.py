"""
Principal Component Analysis (PCA) (NumPy-only).

This module implements a lightweight PCA estimator using SVD, intended for
teaching and small/medium-scale use. The API is scikit-learn-like:
- fit
- transform
- fit_transform
- inverse_transform

Notes
-----
- Centering is always performed.
- Optional whitening is supported.

Example
-------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.pca import PCA
>>> X = np.array([[1., 2.],
...               [3., 4.],
...               [5., 6.]])
>>> pca = PCA(n_components=1, whiten=False).fit(X)
>>> Z = pca.transform(X)
>>> Z.shape
(3, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = ["PCA"]


def _ensure_2d_float(X, name: str = "X") -> np.ndarray:
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


@dataclass
class PCA:
    """
    PCA estimator (NumPy-only).

    Parameters
    ----------
    n_components : int or float or None, default=None
        - If None: keep all components (min(n_samples, n_features)).
        - If int: keep exactly that many components.
        - If float in (0, 1]: keep the smallest number of components such that
          cumulative explained variance ratio >= n_components.
    whiten : bool, default=False
        If True, output components are scaled to have unit variance (approximately).
    eps : float, default=1e-12
        Numerical floor used for whitening.

    Attributes
    ----------
    n_components_ : int
        The selected number of components after fitting.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean used for centering.
    components_ : ndarray of shape (n_components_, n_features)
        Principal axes in feature space.
    singular_values_ : ndarray of shape (n_components_,)
        Singular values corresponding to selected components.
    explained_variance_ : ndarray of shape (n_components_,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Fraction of total variance explained by each selected component.
    """

    n_components: Optional[Union[int, float]] = None
    whiten: bool = False
    eps: float = 1e-12

    # learned
    n_components_: Optional[int] = None
    mean_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None
    singular_values_: Optional[np.ndarray] = None
    explained_variance_: Optional[np.ndarray] = None
    explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X) -> "PCA":
        X = _ensure_2d_float(X, "X")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples to compute variance.")

        if not (np.isfinite(self.eps) and self.eps > 0):
            raise ValueError("eps must be finite and > 0.")

        # center
        mean = X.mean(axis=0)
        Xc = X - mean

        # SVD
        # Xc = U S V^T
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # total variance (ddof=1)
        total_var = np.var(Xc, axis=0, ddof=1).sum()
        if total_var <= 0:
            # all-constant data; components are not meaningful but we can still define a trivial model
            total_var = 0.0

        # explained variance for each component from singular values
        # eigenvalues of covariance = S^2 / (n_samples - 1)
        ev_all = (S**2) / (n_samples - 1)
        if total_var == 0.0:
            evr_all = np.zeros_like(ev_all)
        else:
            evr_all = ev_all / total_var

        k_max = min(n_samples, n_features)

        # choose number of components
        k = self._select_k(evr_all, k_max)

        self.n_components_ = k
        self.mean_ = mean
        self.components_ = Vt[:k].copy()
        self.singular_values_ = S[:k].copy()
        self.explained_variance_ = ev_all[:k].copy()
        self.explained_variance_ratio_ = evr_all[:k].copy()
        return self

    def _select_k(self, evr_all: np.ndarray, k_max: int) -> int:
        nc = self.n_components
        if nc is None:
            return int(k_max)

        if isinstance(nc, (int, np.integer)):
            k = int(nc)
            if k < 1 or k > k_max:
                raise ValueError(f"n_components int must be in [1, {k_max}].")
            return k

        if isinstance(nc, (float, np.floating)):
            r = float(nc)
            if not (0.0 < r <= 1.0):
                raise ValueError("n_components float must be in (0, 1].")
            cumsum = np.cumsum(evr_all[:k_max])
            # smallest k s.t. cumsum >= r
            k = int(np.searchsorted(cumsum, r, side="left") + 1)
            return min(max(k, 1), int(k_max))

        raise TypeError("n_components must be None, int, or float.")

    def _check_is_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None or self.n_components_ is None:
            raise RuntimeError("PCA is not fitted. Call fit(X) first.")

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = _ensure_2d_float(X, "X")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.mean_.shape[0]}.")
        Xc = X - self.mean_
        Z = Xc @ self.components_.T
        if self.whiten:
            denom = np.sqrt(np.maximum(self.explained_variance_, self.eps))
            Z = Z / denom
        return Z

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        self._check_is_fitted()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D; got {Z.ndim}D.")
        if Z.shape[1] != self.n_components_:
            raise ValueError(f"Z has {Z.shape[1]} components, expected {self.n_components_}.")
        Xc = Z
        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, self.eps))
            Xc = Xc * scale
        X_rec = Xc @ self.components_ + self.mean_
        return X_rec
