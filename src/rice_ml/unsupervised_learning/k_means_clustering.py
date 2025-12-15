"""
k-Means clustering (NumPy-only).

This module implements a lightweight k-means estimator with k-means++ initialization.
API is intentionally scikit-learn-like:
- fit
- predict
- fit_predict
- transform (distances to centers)

Notes
-----
- Uses Euclidean distance.
- Handles empty clusters by re-seeding the empty center to the farthest sample.

Example
-------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.k_means_clustering import KMeans

>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [9., 9.],
...               [10., 10.]])
>>> km = KMeans(n_clusters=2, random_state=0).fit(X)
>>> km.cluster_centers_.shape
(2, 2)
>>> labels = km.predict(X)
>>> labels.shape
(4,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

__all__ = ["KMeans"]

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


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distances between all rows of X and centers C.

    Returns
    -------
    D2 : ndarray of shape (n_samples, n_clusters)
        D2[i, j] = ||X[i] - C[j]||^2
    """
    # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(X * X, axis=1, keepdims=True)           # (n, 1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T         # (1, k)
    D2 = x2 + c2 - 2.0 * (X @ C.T)
    return np.maximum(D2, 0.0)


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ initialization.

    Returns
    -------
    centers : ndarray of shape (k, n_features)
    """
    n_samples, n_features = X.shape
    if k < 1 or k > n_samples:
        raise ValueError("k must be in [1, n_samples].")

    centers = np.empty((k, n_features), dtype=float)

    # 1) pick first center uniformly
    idx0 = rng.integers(0, n_samples)
    centers[0] = X[idx0]

    # 2) pick remaining centers with probability proportional to D(x)^2
    # D(x) = distance to nearest chosen center
    closest_d2 = _squared_distances(X, centers[:1]).reshape(-1)

    for i in range(1, k):
        total = closest_d2.sum()
        if not np.isfinite(total) or total <= 0:
            # all points identical to current centers; pick random
            idx = rng.integers(0, n_samples)
            centers[i] = X[idx]
            continue

        probs = closest_d2 / total
        idx = rng.choice(n_samples, p=probs)
        centers[i] = X[idx]

        # update closest distances
        d2_new = _squared_distances(X, centers[i : i + 1]).reshape(-1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    return centers


@dataclass
class KMeans:
    """
    k-Means clustering (NumPy-only).

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {"k-means++", "random"}, default="k-means++"
        Initialization method.
    n_init : int, default=10
        Number of initializations to run; best (lowest inertia) is kept.
    max_iter : int, default=300
        Maximum number of iterations per run.
    tol : float, default=1e-4
        Convergence tolerance on center shift (Frobenius norm).
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels for training data.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run for the best solution.
    """

    n_clusters: int = 8
    init: Literal["k-means++", "random"] = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = None

    # learned
    cluster_centers_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None
    inertia_: Optional[float] = None
    n_iter_: Optional[int] = None

    def fit(self, X: ArrayLike) -> "KMeans":
        X = _ensure_2d_float(X, "X")
        n_samples, n_features = X.shape

        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples.")

        if self.init not in ("k-means++", "random"):
            raise ValueError('init must be "k-means++" or "random".')
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not (isinstance(self.tol, (int, float)) and np.isfinite(self.tol) and self.tol >= 0):
            raise ValueError("tol must be a finite non-negative number.")

        rng = _rng_from_seed(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iter = 0

        # run multiple inits (deterministic under rng stream)
        for _ in range(int(self.n_init)):
            centers0 = self._init_centers(X, rng)
            centers, labels, inertia, n_iter = self._run_lloyd(X, centers0)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_iter)
        return self

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = int(self.n_clusters)
        n_samples = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n_samples, size=k, replace=False)
            return X[idx].astype(float, copy=True)
        return _kmeans_plus_plus_init(X, k, rng).astype(float, copy=False)

    def _run_lloyd(self, X: np.ndarray, centers: np.ndarray):
        k = centers.shape[0]

        prev_centers = centers.copy()
        labels = np.zeros(X.shape[0], dtype=int)

        for it in range(1, int(self.max_iter) + 1):
            # assign
            D2 = _squared_distances(X, centers)
            labels = np.argmin(D2, axis=1)

            # update centers
            new_centers = np.empty_like(centers)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    # empty cluster: re-seed to farthest sample from its nearest center
                    nearest_d2 = np.min(D2, axis=1)
                    idx_far = int(np.argmax(nearest_d2))
                    new_centers[j] = X[idx_far]

            # check convergence
            shift = np.linalg.norm(new_centers - prev_centers)
            centers = new_centers
            if shift <= float(self.tol):
                break
            prev_centers = centers.copy()

        # compute final inertia
        D2_final = _squared_distances(X, centers)
        min_d2 = np.min(D2_final, axis=1)
        inertia = float(np.sum(min_d2))

        return centers, labels, inertia, it

    def _check_is_fitted(self) -> None:
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans is not fitted. Call fit(X) first.")

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X = _ensure_2d_float(X, "X")
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.cluster_centers_.shape[1]}.")
        D2 = _squared_distances(X, self.cluster_centers_)
        return np.argmin(D2, axis=1)

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Return distances to each cluster center.

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
            Euclidean distances from samples to centers.
        """
        self._check_is_fitted()
        X = _ensure_2d_float(X, "X")
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.cluster_centers_.shape[1]}.")
        D2 = _squared_distances(X, self.cluster_centers_)
        return np.sqrt(D2, dtype=float)
