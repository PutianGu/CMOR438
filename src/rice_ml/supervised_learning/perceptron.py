"""
Perceptron classifier (NumPy-only).

This module implements a multiclass Perceptron classifier from scratch using
NumPy only. It supports:
- Binary and multiclass classification (multiclass uses a native multiclass update)
- fit / partial_fit
- decision_function / predict / score
- Optional intercept, shuffling, and early stopping based on mistakes per epoch

Notes
-----
- Perceptron is guaranteed to converge only when the data is linearly separable.
- This implementation is intentionally readable and teaching-oriented, while still
  being a complete and robust model.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.perceptron import PerceptronClassifier
>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [1., 0.],
...               [1., 1.]])
>>> y = np.array([0, 0, 1, 1])
>>> clf = PerceptronClassifier(max_iter=50, random_state=0).fit(X, y)
>>> clf.predict(X).tolist()
[0, 0, 1, 1]
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

__all__ = ["PerceptronClassifier"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
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


def _ensure_1d(y, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


class PerceptronClassifier:
    """
    Perceptron classifier (NumPy-only).

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of epochs over the training data.
    learning_rate : float, default=1.0
        Step size for the perceptron updates.
    fit_intercept : bool, default=True
        If True, learn an intercept term.
    shuffle : bool, default=True
        If True, shuffle samples each epoch.
    tol : int, default=0
        Early stopping tolerance on the change in number of mistakes per epoch.
        If |mistakes_t - mistakes_{t-1}| <= tol for `n_iter_no_change` epochs,
        training stops early.
    n_iter_no_change : int, default=5
        Number of epochs with small improvement to trigger early stopping.
    random_state : int or None, default=None
        Random seed for shuffling and initialization.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels seen during fit.
    coef_ : ndarray of shape (n_classes, n_features) or (1, n_features) for binary
        Weight vectors (without intercept).
    intercept_ : ndarray of shape (n_classes,) or (1,) for binary
        Intercept terms.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Number of epochs actually run.
    mistakes_ : ndarray of shape (n_iter_,)
        Number of mistakes per epoch (training).
    """

    def __init__(
        self,
        max_iter: int = 1000,
        *,
        learning_rate: float = 1.0,
        fit_intercept: bool = True,
        shuffle: bool = True,
        tol: int = 0,
        n_iter_no_change: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.fit_intercept = bool(fit_intercept)
        self.shuffle = bool(shuffle)
        self.tol = int(tol)
        self.n_iter_no_change = int(n_iter_no_change)
        self.random_state = random_state

        # learned
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_iter_: Optional[int] = None
        self.mistakes_: Optional[np.ndarray] = None

        self._W: Optional[np.ndarray] = None  # internal weights incl intercept if enabled
        self._rng: Optional[np.random.Generator] = None

    # ----------------------------- Public API -----------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PerceptronClassifier":
        X_arr, y_arr = self._validate_Xy(X, y)

        self._rng = _rng_from_seed(self.random_state)

        classes = np.unique(y_arr)
        self.classes_ = classes
        n_classes = len(classes)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        y_idx = self._encode_labels(y_arr)

        # weight matrix includes intercept as last column if enabled
        d_aug = n_features + (1 if self.fit_intercept else 0)
        self._W = np.zeros((n_classes, d_aug), dtype=float)

        mistakes_hist = []
        no_change_count = 0
        prev_mistakes = None

        for epoch in range(self.max_iter):
            X_epoch, y_epoch = X_arr, y_idx

            if self.shuffle:
                perm = self._rng.permutation(n_samples)
                X_epoch = X_arr[perm]
                y_epoch = y_idx[perm]

            mistakes = self._run_one_epoch(X_epoch, y_epoch)
            mistakes_hist.append(mistakes)

            # stop if perfect
            if mistakes == 0:
                self.n_iter_ = epoch + 1
                break

            # early stopping on change in mistakes
            if prev_mistakes is not None:
                if abs(prev_mistakes - mistakes) <= self.tol:
                    no_change_count += 1
                else:
                    no_change_count = 0
                if no_change_count >= self.n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break
            prev_mistakes = mistakes
        else:
            self.n_iter_ = self.max_iter

        self.mistakes_ = np.asarray(mistakes_hist, dtype=int)

        # expose sklearn-like coef_/intercept_
        self._sync_public_params()
        return self

    def partial_fit(self, X: ArrayLike, y: ArrayLike, *, classes: Optional[ArrayLike] = None) -> "PerceptronClassifier":
        """
        Incremental training on a batch.

        If called before `fit`, you may pass `classes` to fix the label set.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X_arr.shape[0] == 0:
            raise ValueError("X must be non-empty.")
        if self._rng is None:
            self._rng = _rng_from_seed(self.random_state)

        # initialize if needed
        if self.classes_ is None:
            if classes is None:
                classes = np.unique(y_arr)
            classes_arr = np.unique(np.asarray(classes))
            if classes_arr.size == 0:
                raise ValueError("classes must be non-empty.")
            self.classes_ = classes_arr
            self.n_features_in_ = X_arr.shape[1]
            d_aug = self.n_features_in_ + (1 if self.fit_intercept else 0)
            self._W = np.zeros((len(self.classes_), d_aug), dtype=float)
            self.n_iter_ = 0
            self.mistakes_ = np.asarray([], dtype=int)
        else:
            if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")

        y_idx = self._encode_labels(y_arr)
        mistakes = self._run_one_epoch(X_arr, y_idx)
        self.n_iter_ = int((self.n_iter_ or 0) + 1)
        self.mistakes_ = np.append(self.mistakes_ if self.mistakes_ is not None else np.asarray([], dtype=int), mistakes)

        self._sync_public_params()
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _ensure_2d_float(X, "X")
        if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")

        X_aug = self._augment(X_arr)
        scores = X_aug @ self._W.T  # (n, K)

        # sklearn-like: binary returns shape (n,)
        if scores.shape[1] == 2:
            # use margin between class 1 and class 0
            return scores[:, 1] - scores[:, 0]
        return scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        scores = self.decision_function(X)

        if scores.ndim == 1:
            # binary: scores = margin; predict class1 if margin>0 else class0
            assert self.classes_ is not None
            return np.where(scores > 0, self.classes_[1], self.classes_[0])

        pred_idx = np.argmax(scores, axis=1)
        assert self.classes_ is not None
        return self.classes_[pred_idx]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y lengths do not match.")
        return float(np.mean(y_true == y_pred))

    # ----------------------------- Internals -----------------------------

    def _validate_Xy(self, X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0.")
        if self.n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1.")
        if self.tol < 0:
            raise ValueError("tol must be >= 0.")

        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X_arr.shape[0] == 0:
            raise ValueError("X must be non-empty.")
        return X_arr, y_arr

    def _check_is_fitted(self) -> None:
        if self._W is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        # map labels to indices via searchsorted (classes_ is sorted)
        idx = np.searchsorted(self.classes_, y)
        # ensure all labels are known
        if not np.all(self.classes_[idx] == y):
            raise ValueError("y contains labels not present in classes_.")
        return idx.astype(int, copy=False)

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        ones = np.ones((n, 1), dtype=float)
        return np.hstack([X, ones])

    def _run_one_epoch(self, X: np.ndarray, y_idx: np.ndarray) -> int:
        """Run one epoch of updates; return mistakes count."""
        assert self._W is not None
        X_aug = self._augment(X)

        mistakes = 0
        lr = self.learning_rate

        for i in range(X_aug.shape[0]):
            xi = X_aug[i]
            scores = self._W @ xi  # (K,)
            pred = int(np.argmax(scores))
            true = int(y_idx[i])

            if pred != true:
                # multiclass perceptron update:
                # w_true += lr * x
                # w_pred -= lr * x
                self._W[true] += lr * xi
                self._W[pred] -= lr * xi
                mistakes += 1

        return mistakes

    def _sync_public_params(self) -> None:
        """Expose coef_ and intercept_ like sklearn."""
        assert self._W is not None
        if self.fit_intercept:
            W_no_bias = self._W[:, :-1]
            b = self._W[:, -1]
        else:
            W_no_bias = self._W
            b = np.zeros(self._W.shape[0], dtype=float)

        # sklearn convention: binary has shape (1, n_features)
        if W_no_bias.shape[0] == 2:
            self.coef_ = (W_no_bias[1] - W_no_bias[0]).reshape(1, -1)
            self.intercept_ = np.asarray([b[1] - b[0]], dtype=float)
        else:
            self.coef_ = W_no_bias.copy()
            self.intercept_ = b.copy()
