"""
Community Detection (NumPy-only).

This module provides:
1) Connected Components (CC): baseline partition by graph connectivity.
2) Label Propagation (LP): iterative neighbor-majority label updates.
3) Louvain-style modularity maximization: local moving + graph aggregation.
4) Ground truth helpers: convert an external membership vector to integer labels.

Graphs are provided as adjacency matrices (n_nodes, n_nodes), possibly weighted.

Notes
-----
- LP and Louvain can produce multiple communities even when the graph is fully connected.
- "Ground truth" is not learned from the adjacency; it's a reference partition
  you provide for comparison.

Example (ground-truth helper)
-----------------------------
>>> labels_from_membership(["A", "A", "B", "B", "B"]).tolist()
[0, 0, 1, 1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Literal, Tuple

import numpy as np

__all__ = [
    "labels_from_membership",
    "modularity_score",
    "CommunityDetection",
]

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
Method = Literal["connected_components", "label_propagation", "louvain"]


# ------------------------- Public utilities -------------------------


def labels_from_membership(membership: Sequence) -> np.ndarray:
    """
    Convert an external membership vector (strings/ints/etc.) into labels 0..K-1.

    Parameters
    ----------
    membership : sequence of length n
        Membership labels (e.g., ["Mr. Hi", "Officer", ...]).

    Returns
    -------
    labels : ndarray of shape (n,)
        Integer labels in 0..K-1 (deterministic mapping via sorted unique values).
    """
    arr = np.asarray(membership)
    if arr.ndim != 1:
        raise ValueError("membership must be 1D.")
    if arr.size == 0:
        return np.array([], dtype=int)
    uniq = np.unique(arr)  # sorted unique for determinism
    return np.searchsorted(uniq, arr).astype(int)


def modularity_score(A: ArrayLike, labels: Sequence[int]) -> float:
    """
    Compute modularity Q for an undirected (symmetric) adjacency matrix.

    Uses:
        Q = sum_c [ in_w(c)/m2 - (tot_deg(c)/m2)^2 ]
    where m2 = sum(A) (i.e., 2m in standard notation for undirected graphs).

    Parameters
    ----------
    A : array_like of shape (n, n)
        Adjacency matrix (assumed symmetric).
    labels : sequence of int, length n
        Community assignment.

    Returns
    -------
    float
        Modularity score.
    """
    adj = _ensure_square_adjacency(A)
    n = adj.shape[0]
    lab = np.asarray(labels, dtype=int)

    if lab.ndim != 1 or lab.shape[0] != n:
        raise ValueError("labels must be 1D with length equal to number of nodes.")

    if n == 0:
        return 0.0

    m2 = float(np.sum(adj))
    if m2 == 0.0:
        return 0.0

    k = np.sum(adj, axis=1)
    Q = 0.0
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        if idx.size == 0:
            continue
        in_w = float(np.sum(adj[np.ix_(idx, idx)]))
        tot = float(np.sum(k[idx]))
        Q += (in_w / m2) - (tot / m2) ** 2
    return float(Q)


# ------------------------- Internal helpers -------------------------


def _ensure_square_adjacency(A: ArrayLike) -> np.ndarray:
    arr = np.asarray(A, dtype=float)

    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Adjacency matrix must be 2D; got {arr.ndim}D.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency matrix must be square; got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Adjacency matrix must contain only finite values.")
    return arr


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)
    return np.searchsorted(uniq, labels).astype(int)


def _neighbors_list(adj: np.ndarray) -> list[np.ndarray]:
    n = adj.shape[0]
    out: list[np.ndarray] = []
    for i in range(n):
        neigh = np.flatnonzero(adj[i] != 0.0)
        neigh = neigh[neigh != i]
        out.append(neigh)
    return out


# ------------------------- Connected Components -------------------------


def _connected_components_undirected(adj: np.ndarray) -> np.ndarray:
    """
    Undirected connected components on adjacency matrix.
    Assumes adj is already symmetric if you want undirected semantics.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if float(np.sum(np.abs(adj))) == 0.0:
        return np.arange(n, dtype=int)

    neighs = _neighbors_list(adj)
    labels = -np.ones(n, dtype=int)
    comp_id = 0

    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = comp_id
        while stack:
            v = stack.pop()
            for u in neighs[v]:
                if labels[u] == -1:
                    labels[u] = comp_id
                    stack.append(int(u))
        comp_id += 1

    return labels

def _directed_reachability_components(adj: np.ndarray) -> np.ndarray:
    """
    Directed reachability components:
    For each unvisited node i (in increasing order), mark all nodes reachable
    from i by following outgoing edges as the same component.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if float(np.sum(np.abs(adj))) == 0.0:
        return np.arange(n, dtype=int)

    neighs = _neighbors_list(adj)
    labels = -np.ones(n, dtype=int)
    comp_id = 0

    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = comp_id
        while stack:
            v = stack.pop()
            for u in neighs[v]:
                u = int(u)
                if labels[u] == -1:
                    labels[u] = comp_id
                    stack.append(u)
        comp_id += 1

    return labels

def _strongly_connected_components(adj: np.ndarray) -> np.ndarray:
    """
    Strongly connected components for directed graphs using Kosaraju's algorithm.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if float(np.sum(np.abs(adj))) == 0.0:
        return np.arange(n, dtype=int)

    out_neigh = _neighbors_list(adj)
    in_neigh = _neighbors_list(adj.T)

    visited = np.zeros(n, dtype=bool)
    order: list[int] = []

    # 1st pass: finish-time order on original graph
    for s in range(n):
        if visited[s]:
            continue
        stack: list[tuple[int, int]] = [(s, 0)]
        visited[s] = True
        while stack:
            v, idx = stack[-1]
            neigh = out_neigh[v]
            if idx < neigh.size:
                u = int(neigh[idx])
                stack[-1] = (v, idx + 1)
                if not visited[u]:
                    visited[u] = True
                    stack.append((u, 0))
            else:
                stack.pop()
                order.append(v)

    # 2nd pass: assign components on reversed graph in reverse order
    labels = -np.ones(n, dtype=int)
    comp_id = 0
    for s in reversed(order):
        if labels[s] != -1:
            continue
        stack = [s]
        labels[s] = comp_id
        while stack:
            v = stack.pop()
            for u in in_neigh[v]:
                u = int(u)
                if labels[u] == -1:
                    labels[u] = comp_id
                    stack.append(u)
        comp_id += 1

    return labels


# ------------------------- Label Propagation -------------------------


def _label_propagation(
    adj: np.ndarray,
    *,
    max_iter: int,
    random_state: Optional[int],
    weighted: bool,
) -> Tuple[np.ndarray, int]:
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int), 0

    if float(np.sum(np.abs(adj))) == 0.0:
        return np.arange(n, dtype=int), 0

    rng = np.random.default_rng(None if random_state is None else int(random_state))
    labels = np.arange(n, dtype=int)
    neighs = _neighbors_list(adj)

    for it in range(int(max_iter)):
        changed = 0
        for i in rng.permutation(n):
            neigh = neighs[i]
            if neigh.size == 0:
                continue
            neigh_labels = labels[neigh]

            if weighted:
                w = np.abs(adj[i, neigh]).astype(float, copy=False)
                uniq = np.unique(neigh_labels)
                scores = np.zeros(uniq.shape[0], dtype=float)
                for k, lab in enumerate(uniq):
                    scores[k] = float(np.sum(w[neigh_labels == lab]))
                best = scores.max()
                cand = uniq[scores == best]
                new_label = int(cand.min())
            else:
                uniq, counts = np.unique(neigh_labels, return_counts=True)
                best = counts.max()
                cand = uniq[counts == best]
                new_label = int(cand.min())

            if new_label != labels[i]:
                labels[i] = new_label
                changed += 1

        if changed == 0:
            return labels, it + 1

    return labels, int(max_iter)


# ------------------------- Louvain (modularity) -------------------------


def _aggregate_graph(adj: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    n = adj.shape[0]
    new_adj = np.zeros((k, k), dtype=float)
    rows = np.repeat(labels, n)
    cols = np.tile(labels, n)
    np.add.at(new_adj, (rows, cols), adj.ravel())
    return new_adj


def _louvain_phase1(
    adj: np.ndarray,
    *,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int]:
    n = adj.shape[0]
    labels = np.arange(n, dtype=int)

    if n == 0:
        return labels, 0
    if float(np.sum(np.abs(adj))) == 0.0:
        return labels, 0

    neighs = _neighbors_list(adj)
    cur_Q = modularity_score(adj, labels)

    for it in range(int(max_iter)):
        improved = False
        for i in rng.permutation(n):
            neigh = neighs[i]
            if neigh.size == 0:
                continue

            old = int(labels[i])
            cand_comms = np.unique(labels[neigh])
            if old not in cand_comms:
                cand_comms = np.concatenate([cand_comms, np.array([old], dtype=int)])

            best_comm = old
            best_Q = cur_Q

            for c in cand_comms:
                if int(c) == old:
                    continue
                labels[i] = int(c)
                Q_new = modularity_score(adj, labels)
                if Q_new > best_Q + tol:
                    best_Q = Q_new
                    best_comm = int(c)

            labels[i] = best_comm
            if best_comm != old:
                cur_Q = best_Q
                improved = True

        if not improved:
            return labels, it + 1

    return labels, int(max_iter)


def _louvain(
    adj: np.ndarray,
    *,
    random_state: Optional[int],
    max_levels: int,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int]:
    n0 = adj.shape[0]
    if n0 == 0:
        return np.array([], dtype=int), 0

    rng = np.random.default_rng(None if random_state is None else int(random_state))
    orig_map = np.arange(n0, dtype=int)
    current_adj = adj
    total_iters = 0

    for _level in range(int(max_levels)):
        labels_lvl, iters = _louvain_phase1(current_adj, rng=rng, max_iter=max_iter, tol=tol)
        total_iters += iters
        labels_lvl = _relabel_consecutive(labels_lvl)

        orig_map = labels_lvl[orig_map]
        k = int(np.unique(labels_lvl).size)

        if k == current_adj.shape[0]:
            break

        current_adj = _aggregate_graph(current_adj, labels_lvl, k)

    final_labels = _relabel_consecutive(orig_map)
    return final_labels, total_iters


# ------------------------- Public API -------------------------


@dataclass
class CommunityDetection:
    """
    Community Detection estimator.

    Parameters
    ----------
    method : {"connected_components", "label_propagation", "louvain"}, default="connected_components"
        Which community detection method to use.
    assume_undirected : bool, default=True
        If True, symmetrize adjacency before running algorithms (treat directed as undirected).
        If False and method="connected_components", computes strongly connected components.
    random_state : int or None, default=None
        Seed for reproducibility.
    max_iter : int, default=100
        Max iterations (LP iterations, or Louvain phase-1 iterations per level).
    weighted : bool, default=True
        For label propagation: neighbor votes weighted by edge weights.
    max_levels : int, default=10
        For Louvain: maximum number of aggregation levels.
    tol : float, default=1e-12
        For Louvain: required modularity improvement to accept a move.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Learned labels.
    n_iter_ : int
        Iterations actually run (sum across Louvain levels). For connected_components, equals 1.
    n_communities_ : int
        Number of communities found.
    """

    method: Method = "connected_components"
    assume_undirected: bool = True
    random_state: Optional[int] = None
    max_iter: int = 100

    # LP options
    weighted: bool = True

    # Louvain options
    max_levels: int = 10
    tol: float = 1e-12

    labels_: Optional[np.ndarray] = None
    n_iter_: int = 0
    n_communities_: int = 0

    def __post_init__(self) -> None:
        if self.method not in ("connected_components", "label_propagation", "louvain"):
            raise ValueError(
                "method must be 'connected_components', 'label_propagation', or 'louvain'."
            )
        if not isinstance(self.assume_undirected, bool):
            raise TypeError("assume_undirected must be a bool.")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.random_state is not None and not isinstance(self.random_state, (int, np.integer)):
            raise TypeError("random_state must be an int or None.")
        if not isinstance(self.max_levels, (int, np.integer)) or int(self.max_levels) <= 0:
            raise ValueError("max_levels must be a positive integer.")
        if not (isinstance(self.tol, (int, float)) and np.isfinite(self.tol) and self.tol >= 0):
            raise ValueError("tol must be a finite nonnegative number.")
        if not isinstance(self.weighted, bool):
            raise TypeError("weighted must be a bool.")

    def fit(self, A: ArrayLike) -> "CommunityDetection":
        adj = _ensure_square_adjacency(A)

        # Optional symmetrization (treat directed as undirected)
        if self.assume_undirected and adj.size > 0 and not np.allclose(adj, adj.T, atol=1e-12, rtol=0.0):
            adj = 0.5 * (adj + adj.T)

        if self.method == "connected_components":
            if self.assume_undirected:
                labels = _connected_components_undirected(adj)
            else:
                labels = _directed_reachability_components(adj)
            iters = 1

        elif self.method == "label_propagation":
            labels, iters = _label_propagation(
                adj,
                max_iter=int(self.max_iter),
                random_state=self.random_state,
                weighted=self.weighted,
            )

        else:  # louvain
            labels, iters = _louvain(
                adj,
                random_state=self.random_state,
                max_levels=int(self.max_levels),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
            )

        labels = _relabel_consecutive(np.asarray(labels, dtype=int))
        self.labels_ = labels
        self.n_iter_ = int(iters)
        self.n_communities_ = int(np.unique(labels).size)
        return self

    def fit_predict(self, A: ArrayLike) -> np.ndarray:
        self.fit(A)
        assert self.labels_ is not None
        return self.labels_


