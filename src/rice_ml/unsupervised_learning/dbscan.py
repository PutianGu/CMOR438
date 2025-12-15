"""
DBSCAN clustering (NumPy-only).

DBSCAN groups points that are closely packed together, marking points that lie
alone in low-density regions as outliers.

Definitions
----------
- eps: neighborhood radius
- min_samples: minimum number of points (including itself) required to form a core point
- core point: |N_eps(p)| >= min_samples
- border point: not core, but within eps of a core point
- noise: neither core nor border (label = -1)

API
---
- fit
- fit_predict

Notes
-----
- This implementation uses Euclidean distance and is intended for teaching.
- Time complexity is O(n^2) due to distance computations.

Example
-------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.dbscan import DBSCAN
>>> X = np.array([[0., 0.],
...               [0., 0.1],
...               [0.1, 0.],
...               [10., 10.],
...               [10.1, 10.],
...               [50., 50.]])
>>> labels = DBSCAN(eps=0.25, min_samples=2).fit_predict(X)
>>> labels.shape
(6,)
>>> set(labels.tolist()) >= {-1, 0, 1}
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["DBSCAN"]

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
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")
    return arr


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances for X.

    Returns
    -------
    D2 : ndarray of shape (n, n)
        D2[i, j] = ||X[i] - X[j]||^2
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x2 = np.sum(X * X, axis=1, keepdims=True)  # (n, 1)
    D2 = x2 + x2.T - 2.0 * (X @ X.T)
    return np.maximum(D2, 0.0)


@dataclass
class DBSCAN:
    """
    DBSCAN clustering (NumPy-only).

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius.
    min_samples : int, default=5
        Minimum number of points in eps-neighborhood (including itself)
        for a point to be considered a core point.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Noise is labeled -1.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core points in the dataset.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    """

    eps: float = 0.5
    min_samples: int = 5

    labels_: Optional[np.ndarray] = None
    core_sample_indices_: Optional[np.ndarray] = None
    n_clusters_: Optional[int] = None

    def fit(self, X: ArrayLike) -> "DBSCAN":
        X = _ensure_2d_float(X, "X")
        n = X.shape[0]

        if not (isinstance(self.eps, (int, float)) and np.isfinite(self.eps) and self.eps > 0):
            raise ValueError("eps must be a finite number > 0.")
        if not isinstance(self.min_samples, (int, np.integer)) or int(self.min_samples) < 1:
            raise ValueError("min_samples must be an integer >= 1.")
        min_samples = int(self.min_samples)

        # Precompute eps-neighborhood adjacency (n x n)
        eps2 = float(self.eps) ** 2
        D2 = _pairwise_squared_distances(X)
        neigh = D2 <= eps2  # boolean adjacency

        # Core points: at least min_samples neighbors (including itself)
        neigh_counts = neigh.sum(axis=1)
        is_core = neigh_counts >= min_samples

        labels = np.full(n, -1, dtype=int)  # -1 = noise initially
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            if not is_core[i]:
                # border/noise for now; might be relabeled later when reached from core
                continue

            # start a new cluster
            labels[i] = cluster_id
            # seed set: all neighbors of i
            seeds = list(np.flatnonzero(neigh[i]))

            # expand cluster
            j = 0
            while j < len(seeds):
                p = seeds[j]
                if not visited[p]:
                    visited[p] = True
                    if is_core[p]:
                        # add its neighbors to the seed list
                        p_neighbors = np.flatnonzero(neigh[p])
                        # append only new ones (avoid O(n^2) duplicates blowup)
                        # we do a simple membership check via a boolean mask built once per loop
                        # for teaching simplicity, keep it readable:
                        for q in p_neighbors:
                            if q not in seeds:
                                seeds.append(int(q))
                # assign to cluster if not already assigned
                if labels[p] == -1:
                    labels[p] = cluster_id
                j += 1

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        self.n_clusters_ = int(cluster_id)
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).labels_
