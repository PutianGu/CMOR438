from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class StandardScaler:
    """
    简单版标准化器：按列减均值除标准差。
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: ArrayLike):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        # 避免除以 0
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: ArrayLike):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: ArrayLike):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        X_scaled = np.asarray(X_scaled, dtype=float)
        return X_scaled * self.scale_ + self.mean_
