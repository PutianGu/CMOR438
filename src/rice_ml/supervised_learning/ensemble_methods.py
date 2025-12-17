"""
Ensemble methods for the rice_ml package (NumPy-only).

This module provides simple, educational implementations of bagging-based
ensembles using the existing decision tree and regression tree classes.

Currently implemented
---------------------
RandomForestClassifier
RandomForestRegressor
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .decision_trees import DecisionTreeClassifier
from .regression_trees import RegressionTreeRegressor

__all__ = ["RandomForestClassifier", "RandomForestRegressor"]

MaxFeatures = Union[None, int, float, str]


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> Union[None, int, float]:
    """Convert common string shorthands to int, otherwise pass through."""
    if max_features is None:
        return None
    if isinstance(max_features, (int, float)):
        return max_features
    if isinstance(max_features, str):
        mf = max_features.lower().strip()
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features))) if n_features > 1 else 1
        raise ValueError("max_features string must be one of: {'sqrt', 'log2'}.")
    raise ValueError("max_features must be None, int, float, or a supported string.")


class RandomForestClassifier:
    """Random forest classifier (bagged decision trees)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: List[DecisionTreeClassifier] = []
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive int.")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty.")
        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values.")

        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_.clear()

        resolved_max_features = _resolve_max_features(self.max_features, self.n_features_)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = self._rng.integers(0, X.shape[0], size=X.shape[0])
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            tree_seed = int(self._rng.integers(0, 2**31 - 1))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=resolved_max_features,
                random_state=tree_seed,
            )
            tree.fit(Xb, yb)
            self.estimators_.append(tree)

        self.n_classes_ = self.estimators_[0].n_classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_ or self.n_classes_ is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}.")
        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values.")

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for est in self.estimators_:
            proba += est.predict_proba(X)
        proba /= len(self.estimators_)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class RandomForestRegressor:
    """Random forest regressor (bagged regression trees)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: List[RegressionTreeRegressor] = []
        self.n_features_: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X = np.asarray(X)
        y = np.asarray(y)

        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive int.")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of numeric targets.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty.")

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must be numeric for regression.")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("X and y must not contain NaN values.")

        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_.clear()

        resolved_max_features = _resolve_max_features(self.max_features, self.n_features_)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = self._rng.integers(0, X.shape[0], size=X.shape[0])
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            tree_seed = int(self._rng.integers(0, 2**31 - 1))
            tree = RegressionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=resolved_max_features,
                random_state=tree_seed,
            )
            tree.fit(Xb, yb)
            self.estimators_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}.")
        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values.")

        preds = np.zeros(X.shape[0], dtype=float)
        for est in self.estimators_:
            preds += est.predict(X)
        preds /= len(self.estimators_)
        return preds
