from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels=None) -> np.ndarray:
    """
    简单版混淆矩阵实现。

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    labels : list or array of labels, optional
        若为 None，则从 y_true 与 y_pred 中自动推断，并按升序排序。

    Returns
    -------
    cm : ndarray, shape (n_classes, n_classes)
        cm[i, j] = 真实类 labels[i] 被预测为 labels[j] 的样本数量。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    L = len(labels)
    index = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1

    return cm
