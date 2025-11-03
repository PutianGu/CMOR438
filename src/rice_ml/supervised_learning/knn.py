from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .functions import pairwise_distance


class KNNClassifier:
    """
    Minimal KNN classifier.

    Parameters
    ----------
    n_neighbors : int
    metric : {'euclidean', 'manhattan'}
    weights : {'uniform', 'distance'}
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be >= 1")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")

        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self._X = X
        self._y = y
        self.classes_ = np.unique(y)
        return self

    def _vote(self, neighbor_labels: np.ndarray, neighbor_dist: np.ndarray):
        """
        neighbor_labels: (k,)
        neighbor_dist:   (k,)  已按距离从小到大排序
        """
        if self.weights == "uniform":
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            best = np.flatnonzero(counts == counts.max())
            # tie-break: 取较小的标签
            return int(vals[best].min())
        else:  # distance 权重
            weights = 1.0 / (neighbor_dist + 1e-12)
            score: dict[int, float] = {}
            for lbl, w in zip(neighbor_labels, weights):
                score[lbl] = score.get(lbl, 0.0) + float(w)
            best_val = max(score.values())
            best_labels = [k for k, v in score.items() if v == best_val]
            return int(min(best_labels))

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self._X is None or self._y is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        D = pairwise_distance(X, self._X, metric=self.metric)  # (n_test, n_train)

        n_train = self._X.shape[0]
        k = min(self.n_neighbors, n_train)

        # 先找出每个测试点距离最近的 k 个索引
        idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]

        preds: list[int] = []
        for i in range(X.shape[0]):
            neigh_idx = idx[i]                  # (k,)
            neigh_labels = self._y[neigh_idx]   # (k,)
            neigh_dist = D[i, neigh_idx]        # (k,)
            order = np.argsort(neigh_dist)
            pred = self._vote(neigh_labels[order], neigh_dist[order])
            preds.append(pred)

        return np.asarray(preds)
