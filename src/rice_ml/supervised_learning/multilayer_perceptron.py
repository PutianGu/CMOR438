"""
Multilayer perceptron (MLP) classifier for the rice_ml package.

This module implements a small, educational, NumPy-only feed-forward neural
network classifier trained with backpropagation and mini-batch optimization.

Key features
------------
- Fully-connected MLP with configurable hidden layers
- Activations: relu, tanh
- Solvers: sgd, adam
- L2 regularization (weight decay)
- Reproducible training via random_state
- predict_proba / predict (sklearn-like)

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.multilayer_perceptron import MultilayerPerceptronClassifier
>>>
>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [1., 0.],
...               [1., 1.]])
>>> y = np.array([0, 1, 1, 0])  # XOR
>>> clf = MultilayerPerceptronClassifier(hidden_layer_sizes=(8,), activation="tanh",
...                                      solver="adam", learning_rate=0.05,
...                                      max_iter=800, random_state=0)
>>> _ = clf.fit(X, y)
>>> clf.predict(X)
array([0, 1, 1, 0])
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["MultilayerPerceptronClassifier", "MLPClassifier"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((y.size, n_classes), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(np.floor(val_fraction * n)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size == 0:
        # fallback: if too small, just no validation split
        return X, y, X[:0], y[:0]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


class MultilayerPerceptronClassifier:
    """Feed-forward MLP classifier trained with backprop.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...], default=(100,)
        Sizes of hidden layers.
    activation : {"relu", "tanh"}, default="relu"
        Activation function for hidden layers.
    solver : {"sgd", "adam"}, default="adam"
        Optimization algorithm.
    learning_rate : float, default=0.001
        Step size for updates.
    batch_size : int, default=32
        Mini-batch size.
    max_iter : int, default=200
        Number of epochs.
    alpha : float, default=0.0
        L2 regularization strength (weight decay).
    tol : float, default=1e-6
        Minimum improvement in monitored loss to reset patience.
    n_iter_no_change : int, default=10
        Early-stopping patience.
    early_stopping : bool, default=False
        If True, monitor validation loss and stop early.
    validation_fraction : float, default=0.1
        Fraction of training data used for validation when early_stopping=True.
    random_state : int or None, default=None
        RNG seed.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels seen during fit (original label space).
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of input features.
    coefs_ : list[np.ndarray]
        Weight matrices for each layer.
    intercepts_ : list[np.ndarray]
        Bias vectors for each layer.
    loss_curve_ : list[float]
        Training loss per epoch.
    n_iter_ : int
        Number of epochs actually run.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_iter: int = 200,
        alpha: float = 0.0,
        tol: float = 1e-6,
        n_iter_no_change: int = 10,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.coefs_: List[np.ndarray] = []
        self.intercepts_: List[np.ndarray] = []
        self.loss_curve_: List[float] = []
        self.n_iter_: int = 0

        self._rng: Optional[np.random.Generator] = None

        # Adam state
        self._mW: List[np.ndarray] = []
        self._vW: List[np.ndarray] = []
        self._mb: List[np.ndarray] = []
        self._vb: List[np.ndarray] = []
        self._t: int = 0

    # ---------------------------- Public API ----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptronClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_hyperparams()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty.")
        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values.")

        # Encode labels to 0..K-1, but keep original classes_
        classes, y_enc = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        self.n_features_in_ = int(X.shape[1])

        if self.n_classes_ < 2:
            raise ValueError("y must contain at least 2 classes for classification.")

        self._rng = np.random.default_rng(self.random_state)

        # Optional validation split (for early stopping)
        if self.early_stopping:
            X_tr, y_tr, X_val, y_val = _train_val_split(
                X, y_enc, self.validation_fraction, self._rng
            )
        else:
            X_tr, y_tr, X_val, y_val = X, y_enc, X[:0], y_enc[:0]

        # Initialize parameters
        layer_sizes = [self.n_features_in_, *list(self.hidden_layer_sizes), self.n_classes_]
        self._init_params(layer_sizes)

        self.loss_curve_.clear()
        self.n_iter_ = 0

        best_val_loss = np.inf
        best_params = None
        no_improve = 0

        for epoch in range(1, self.max_iter + 1):
            self.n_iter_ = epoch

            # Shuffle
            idx = self._rng.permutation(X_tr.shape[0])
            X_shuf = X_tr[idx]
            y_shuf = y_tr[idx]

            # Mini-batches
            for start in range(0, X_shuf.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_shuf.shape[0])
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                self._train_step(Xb, yb)

            # Monitor training loss (on full training split)
            train_loss = self._loss(X_tr, y_tr)
            self.loss_curve_.append(train_loss)

            if self.early_stopping and X_val.shape[0] > 0:
                val_loss = self._loss(X_val, y_val)

                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    best_params = (self._copy_params(), self._copy_adam_state(), self._t)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        # Restore best params before stopping
                        if best_params is not None:
                            (Wb, bb), (mW, vW, mb, vb), t = best_params
                            self.coefs_, self.intercepts_ = Wb, bb
                            self._mW, self._vW, self._mb, self._vb = mW, vW, mb, vb
                            self._t = t
                        break
            else:
                # simple patience on training loss if no validation split
                if epoch > 1 and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self.n_iter_no_change:
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values.")

        logits = self._forward_logits(X)
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        return self.classes_[pred_idx]

    # ---------------------------- Core math -----------------------------
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (activations, pre-activations) for all layers."""
        A: List[np.ndarray] = [X]
        Z: List[np.ndarray] = []

        for i in range(len(self.coefs_) - 1):
            z = A[-1] @ self.coefs_[i] + self.intercepts_[i]
            a = self._activate(z)
            Z.append(z)
            A.append(a)

        # output logits
        z_out = A[-1] @ self.coefs_[-1] + self.intercepts_[-1]
        Z.append(z_out)
        A.append(z_out)  # keep logits as last "activation" for convenience
        return A, Z

    def _forward_logits(self, X: np.ndarray) -> np.ndarray:
        A = X
        for i in range(len(self.coefs_) - 1):
            A = self._activate(A @ self.coefs_[i] + self.intercepts_[i])
        return A @ self.coefs_[-1] + self.intercepts_[-1]

    def _loss(self, X: np.ndarray, y_enc: np.ndarray) -> float:
        logits = self._forward_logits(X)
        P = _softmax(logits)
        # cross entropy
        eps = 1e-12
        ce = -np.mean(np.log(P[np.arange(y_enc.size), y_enc] + eps))
        # L2 on weights (not biases)
        l2 = 0.0
        if self.alpha > 0.0:
            l2 = 0.5 * self.alpha * sum(float(np.sum(W * W)) for W in self.coefs_) / X.shape[0]
        return float(ce + l2)

    def _train_step(self, Xb: np.ndarray, yb: np.ndarray) -> None:
        # Forward
        A, Z = self._forward(Xb)
        logits = Z[-1]
        P = _softmax(logits)

        n = Xb.shape[0]
        Y = _one_hot(yb, self.n_classes_)  # type: ignore[arg-type]

        # dL/dlogits for softmax+CE
        dZ = (P - Y) / n

        # Gradients containers
        dW: List[np.ndarray] = [np.zeros_like(W) for W in self.coefs_]
        db: List[np.ndarray] = [np.zeros_like(b) for b in self.intercepts_]

        # Output layer grads
        dW[-1] = A[-2].T @ dZ
        db[-1] = np.sum(dZ, axis=0)

        # Backprop through hidden layers
        dA_prev = dZ @ self.coefs_[-1].T
        for layer in range(len(self.coefs_) - 2, -1, -1):
            dZ_h = dA_prev * self._activate_derivative(Z[layer])
            dW[layer] = A[layer].T @ dZ_h
            db[layer] = np.sum(dZ_h, axis=0)
            if layer > 0:
                dA_prev = dZ_h @ self.coefs_[layer].T

        # L2 regularization on weights
        if self.alpha > 0.0:
            for i in range(len(dW)):
                dW[i] += self.alpha * self.coefs_[i]

        # Update
        if self.solver == "sgd":
            for i in range(len(self.coefs_)):
                self.coefs_[i] -= self.learning_rate * dW[i]
                self.intercepts_[i] -= self.learning_rate * db[i]
        else:
            self._adam_update(dW, db)

    # ---------------------------- Optimizer -----------------------------
    def _adam_update(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        # Adam hyperparameters (fixed for simplicity)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        self._t += 1
        t = self._t

        for i in range(len(self.coefs_)):
            self._mW[i] = beta1 * self._mW[i] + (1.0 - beta1) * dW[i]
            self._vW[i] = beta2 * self._vW[i] + (1.0 - beta2) * (dW[i] ** 2)
            self._mb[i] = beta1 * self._mb[i] + (1.0 - beta1) * db[i]
            self._vb[i] = beta2 * self._vb[i] + (1.0 - beta2) * (db[i] ** 2)

            mW_hat = self._mW[i] / (1.0 - beta1 ** t)
            vW_hat = self._vW[i] / (1.0 - beta2 ** t)
            mb_hat = self._mb[i] / (1.0 - beta1 ** t)
            vb_hat = self._vb[i] / (1.0 - beta2 ** t)

            self.coefs_[i] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            self.intercepts_[i] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

    # ---------------------------- Activation ----------------------------
    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, z)
        # tanh
        return np.tanh(z)

    def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0.0).astype(float)
        # tanh'
        t = np.tanh(z)
        return 1.0 - t * t

    # ---------------------------- Init & checks -------------------------
    def _init_params(self, layer_sizes: Sequence[int]) -> None:
        assert self._rng is not None
        self.coefs_.clear()
        self.intercepts_.clear()

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.activation == "relu":
                # He init
                W = self._rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
            else:
                # Xavier init (tanh)
                W = self._rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, out_dim))
            b = np.zeros(out_dim, dtype=float)
            self.coefs_.append(W.astype(float))
            self.intercepts_.append(b)

        # Adam state init
        self._mW = [np.zeros_like(W) for W in self.coefs_]
        self._vW = [np.zeros_like(W) for W in self.coefs_]
        self._mb = [np.zeros_like(b) for b in self.intercepts_]
        self._vb = [np.zeros_like(b) for b in self.intercepts_]
        self._t = 0

    def _validate_hyperparams(self) -> None:
        if not isinstance(self.hidden_layer_sizes, tuple) or len(self.hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes must be a non-empty tuple of positive ints.")
        if any((not isinstance(h, int) or h <= 0) for h in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive ints.")

        if self.activation not in {"relu", "tanh"}:
            raise ValueError("activation must be one of {'relu', 'tanh'}.")

        if self.solver not in {"sgd", "adam"}:
            raise ValueError("solver must be one of {'sgd', 'adam'}.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive int.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive int.")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(self.alpha, (int, float)) or self.alpha < 0:
            raise ValueError("alpha must be a non-negative float.")
        if not isinstance(self.tol, (int, float)) or self.tol < 0:
            raise ValueError("tol must be a non-negative float.")
        if not isinstance(self.n_iter_no_change, int) or self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be a positive int.")

        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be a bool.")
        if not (0.0 < float(self.validation_fraction) < 0.5):
            raise ValueError("validation_fraction must be in (0, 0.5).")

    def _check_is_fitted(self) -> None:
        if self.n_features_in_ is None or self.n_classes_ is None or not self.coefs_:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")

    def _copy_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        Wb = [W.copy() for W in self.coefs_]
        bb = [b.copy() for b in self.intercepts_]
        return Wb, bb

    def _copy_adam_state(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        mW = [x.copy() for x in self._mW]
        vW = [x.copy() for x in self._vW]
        mb = [x.copy() for x in self._mb]
        vb = [x.copy() for x in self._vb]
        return mW, vW, mb, vb


# Friendly alias
MLPClassifier = MultilayerPerceptronClassifier
