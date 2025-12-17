"""
Regression tree regressor implementation for the rice_ml package.

This module provides a CART-style regression tree implemented from scratch
using NumPy only. Splits are chosen by minimizing the weighted variance
(i.e., MSE/SSE criterion). The API is intentionally sklearn-like.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.regression_trees import RegressionTreeRegressor
>>>
>>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
>>> y = np.array([0.0, 0.0, 1.0, 1.0])
>>> tree = RegressionTreeRegressor(max_depth=1, random_state=0)
>>> _ = tree.fit(X, y)
>>> tree.predict(X)
array([0., 0., 1., 1.])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["RegressionTreeRegressor"]


@dataclass
class _TreeNode:
    """Internal node representation for a regression tree."""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    value: Optional[float] = None  # leaf prediction = mean(y) at that node

    def is_leaf(self) -> bool:
        return self.feature_index is None


class RegressionTreeRegressor:
    """CART-style regression tree using variance/MSE reduction.

    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. If None, grow until stopping conditions.
    min_samples_split : int, default=2
        Minimum samples required to attempt a split.
    min_samples_leaf : int, default=1
        Minimum samples required in each leaf.
    max_features : int or float or None, optional
        Features to consider per split.
        - None: use all features
        - int: use exactly that many features (1..n_features)
        - float in (0,1]: use that fraction of n_features (at least 1)
    random_state : int or None, optional
        RNG seed used for feature subsampling.

    Attributes
    ----------
    n_features_ : int
        Number of features seen during fit.
    tree_ : _TreeNode
        Root node of the fitted regression tree.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float | int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_: Optional[int] = None
        self.tree_: Optional[_TreeNode] = None
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTreeRegressor":
        """Fit the regression tree on training data."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of numeric targets.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty.")

        if self.max_depth is not None:
            if not isinstance(self.max_depth, int) or self.max_depth <= 0:
                raise ValueError("max_depth must be a positive int or None.")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an int >= 2.")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an int >= 1.")
        if self.min_samples_leaf * 2 > X.shape[0] and self.min_samples_split <= X.shape[0]:
            # not an error, but it effectively prevents any split
            pass

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must be numeric for regression.")

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("X and y must not contain NaN values.")

        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous targets for samples in X."""
        if self.tree_ is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}.")

        preds = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            node = self._traverse_tree(X[i], self.tree_)
            if node.value is None:
                raise RuntimeError("Internal error: reached leaf without a value.")
            preds[i] = node.value
        return preds

    # ------------------------------------------------------------------
    # Tree growth
    # ------------------------------------------------------------------
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n_samples = X.shape[0]
        value = float(np.mean(y)) if n_samples > 0 else 0.0

        # stopping: pure (zero variance), max depth, or too few samples
        if (
            self._variance(y) == 0.0
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return _TreeNode(value=value)

        feat_idx, threshold, (left_mask, right_mask) = self._best_split(X, y)

        if feat_idx is None:
            return _TreeNode(value=value)

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature_index=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
            value=value,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, (np.array([], dtype=bool), np.array([], dtype=bool))

        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)

        feature_indices = self._choose_features(n_features)

        best_score = np.inf
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_left = np.array([], dtype=bool)
        best_right = np.array([], dtype=bool)

        for feat in feature_indices:
            x_col = X[:, feat]
            uniq = np.unique(x_col)

            # no split possible
            if uniq.size <= 1:
                continue

            # Use midpoints between sorted unique values (more standard for continuous features)
            uniq_sorted = np.sort(uniq)
            thresholds = (uniq_sorted[:-1] + uniq_sorted[1:]) / 2.0

            for thr in thresholds:
                left_mask = x_col <= thr
                right_mask = ~left_mask

                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # weighted variance (equivalent to minimizing SSE / MSE)
                score = (n_left * self._variance(y[left_mask]) + n_right * self._variance(y[right_mask])) / n_samples

                if score < best_score:
                    best_score = score
                    best_feat = int(feat)
                    best_thresh = float(thr)
                    best_left = left_mask
                    best_right = right_mask

        if best_feat is None:
            return None, None, (np.array([], dtype=bool), np.array([], dtype=bool))

        return best_feat, best_thresh, (best_left, best_right)

    def _choose_features(self, n_features: int) -> np.ndarray:
        """Choose feature indices based on max_features setting."""
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features int must be in [1, n_features].")
            assert self._rng is not None
            return self._rng.choice(n_features, self.max_features, replace=False)

        if isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
            assert self._rng is not None
            return self._rng.choice(n_features, k, replace=False)

        raise ValueError("max_features must be None, int, or float.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _variance(y: np.ndarray) -> float:
        """Population variance (ddof=0)."""
        if y.size == 0:
            return 0.0
        return float(np.var(y))

    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> _TreeNode:
        """Traverse until leaf."""
        while not node.is_leaf():
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node
