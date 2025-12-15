"""
Community detection (NumPy-only).

This module provides a small, teaching-friendly community detection estimator
for graph data represented as an adjacency matrix.

Supported methods
-----------------
- "connected_components" (default): communities are connected components of the graph.
- "label_propagation": simple label propagation (optional; deterministic tie-breaks).

Input format
------------
Adjacency matrix A of shape (n_nodes, n_nodes). Nonzero entries indicate edges.
Works for unweighted/weighted graphs. By default, treats the graph as undirected.

Example
-------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.community_detection import CommunityDetection
>>> A = np.array([
...     [0, 1, 0, 0],
...     [1, 0, 1, 0],
...     [0, 1, 0, 0],
...     [0, 0, 0, 0],
... ], dtype=float)
>>> cd = CommunityDetection(method="connected_components")
>>> labels = cd.fit_predict(A)
>>> labels.shape
(4,)
>>> set(labels.tolist()) >= {0, 1}
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

__all__ = ["CommunityDetection"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_square_numeric(A: ArrayLike, name: str = "A") -> np.ndarray:
    arr = np.asarray(A)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got {arr.ndim}D.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square; got shape {arr.shape}.")
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


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _to_undirected_bool_adj(A: np.ndarray, *, assume_undirected: bool) -> np.ndarray:
    """
    Convert numeric adjacency to boolean adjacency.
    Nonzero entries indicate edges. Optionally symmetrize.

    Returns
    -------
    adj : ndarray bool shape (n, n)
    """
    adj = (A != 0)
    if assume_undirected:
        adj = np.logical_or(adj, adj.T)
    # Remove self-loop effect for traversal (doesn't matter much, but cleaner)
    np.fill_diagonal(adj, False)
    return adj


def _connected_components(adj: np.ndarray) -> np.ndarray:
    """
    Connected components labeling for a boolean adjacency matrix.
    Returns labels 0..(n_components-1).
    """
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=int)
    cid = 0

    for start in range(n):
        if labels[start] != -1:
            continue
        # BFS/DFS
        stack = [start]
        labels[start] = cid
        while stack:
            v = stack.pop()
            neigh = np.flatnonzero(adj[v])
            for u in neigh:
                if labels[u] == -1:
                    labels[u] = cid
                    stack.append(int(u))
        cid += 1
    return labels


def _label_propagation(
    adj: np.ndarray,
    *,
    max_iter: int,
    random_state: Optional[int],
) -> np.ndarray:
    """
    Simple label propagation algorithm (LPA).
    - Start with unique label per node.
    - Iteratively update a node's label to the most frequent label among neighbors.
    - Deterministic tie-break: choose the smallest label among maxima.
    - Node update order is randomized if random_state is set; otherwise default RNG.

    Returns
    -------
    labels : ndarray shape (n,)
    """
    n = adj.shape[0]
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    rng = _rng_from_seed(random_state)

    labels = np.arange(n, dtype=int)
    order = np.arange(n, dtype=int)

    for _ in range(max_iter):
        changed = 0
        rng.shuffle(order)
        for v in order:
            neigh = np.flatnonzero(adj[v])
            if neigh.size == 0:
                continue
            neigh_labels = labels[neigh]
            # most frequent label, tie -> smallest label
            vals, counts = np.unique(neigh_labels, return_counts=True)
            best_count = counts.max()
            best_vals = vals[counts == best_count]
            new_label = int(best_vals.min())
            if new_label != labels[v]:
                labels[v] = new_label
                changed += 1
        if changed == 0:
            break

    # compress labels to 0..K-1 for consistency
    uniq = np.unique(labels)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    return np.array([mapping[int(l)] for l in labels], dtype=int)


@dataclass
class CommunityDetection:
    """
    Community detection for graphs given an adjacency matrix.

    Parameters
    ----------
    method : {"connected_components", "label_propagation"}, default="connected_components"
        Which algorithm to use.
    assume_undirected : bool, default=True
        If True, treats the graph as undirected by symmetrizing adjacency.
    max_iter : int, default=50
        Only used for method="label_propagation".
    random_state : int or None, default=None
        Only used for method="label_propagation" to randomize update order.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Community labels (0..K-1).
    n_communities_ : int
        Number of detected communities.
    """

    method: Literal["connected_components", "label_propagation"] = "connected_components"
    assume_undirected: bool = True
    max_iter: int = 50
    random_state: Optional[int] = None

    labels_: Optional[np.ndarray] = None
    n_communities_: Optional[int] = None

    def fit(self, A: ArrayLike) -> "CommunityDetection":
        A = _ensure_square_numeric(A, "A")

        if self.method not in ("connected_components", "label_propagation"):
            raise ValueError('method must be "connected_components" or "label_propagation".')
        if not isinstance(self.assume_undirected, (bool, np.bool_)):
            raise TypeError("assume_undirected must be a bool.")
        if not isinstance(self.max_iter, (int, np.integer)):
            raise TypeError("max_iter must be an int.")

        adj = _to_undirected_bool_adj(A, assume_undirected=bool(self.assume_undirected))

        if self.method == "connected_components":
            labels = _connected_components(adj)
        else:
            labels = _label_propagation(adj, max_iter=int(self.max_iter), random_state=self.random_state)

        self.labels_ = labels
        self.n_communities_ = int(np.unique(labels).size)
        return self

    def fit_predict(self, A: ArrayLike) -> np.ndarray:
        return self.fit(A).labels_
