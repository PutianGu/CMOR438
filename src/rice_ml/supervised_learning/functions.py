from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def euclidean(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance between two 1D points."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b))


def manhattan(a: ArrayLike, b: ArrayLike) -> float:
    """Manhattan (L1) distance between two 1D points."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).sum())


def pairwise_distance(X1: ArrayLike, X2: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distances between rows of X1 and X2.

    Parameters
    ----------
    X1 : array-like, shape (n_samples_1, n_features)
    X2 : array-like, shape (n_samples_2, n_features)
    metric : {'euclidean', 'manhattan'}

    Returns
    -------
    D : ndarray, shape (n_samples_1, n_samples_2)
        D[i, j] = distance between X1[i] and X2[j]
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    if metric not in {"euclidean", "manhattan"}:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "euclidean":
        # (x - y)^2 = x^2 + y^2 - 2 x·y
        X1_sq = (X1 ** 2).sum(axis=1, keepdims=True)        # (n1, 1)
        X2_sq = (X2 ** 2).sum(axis=1, keepdims=True).T      # (1, n2)
        D2 = X1_sq + X2_sq - 2 * X1 @ X2.T                  # (n1, n2)
        np.maximum(D2, 0, out=D2)                           # 数值稳定
        return np.sqrt(D2)
    else:  # metric == "manhattan"
        return np.abs(X1[:, None, :] - X2[None, :, :]).sum(axis=2)
