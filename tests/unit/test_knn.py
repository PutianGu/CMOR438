import numpy as np
import pytest
from rice_ml.supervised_learning.knn import KNNClassifier


def make_toy():
    # 两团可分数据
    X0 = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
    X1 = np.array([[5, 5], [5, 6], [6, 5]], dtype=float)
    X = np.vstack([X0, X1])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# ---- 基础正确性 ----
def test_knn_k1():
    X, y = make_toy()
    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)
    pred = model.predict(np.array([[0.2, 0.1], [5.2, 5.1]]))
    assert (pred == np.array([0, 1])).all()


def test_knn_k3_distance_weighted():
    X, y = make_toy()
    model = KNNClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)
    pred = model.predict(np.array([[0.3, 0.1], [5.4, 5.4]]))
    assert (pred == np.array([0, 1])).all()


# ---- 边界与鲁棒性 ----
def test_invalid_params():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=3, metric="chebyshev")
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=3, weights="weird")


def test_shape_and_length_mismatch():
    model = KNNClassifier()
    X_bad = np.array([1.0, 2.0, 3.0])  # 1D
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError):
        model.fit(X_bad, y)

    X2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    y2 = np.array([0])
    with pytest.raises(ValueError):
        model.fit(X2, y2)


def test_predict_before_fit():
    model = KNNClassifier()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.0, 0.0]]))


def test_k_greater_than_n_train():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = KNNClassifier(n_neighbors=5)  # k > n_train
    model.fit(X, y)
    pred = model.predict(np.array([[0.1, 0.0]]))
    assert pred.shape == (1,)


def test_tie_break_uniform():
    X = np.array([[-1.0, 0.0], [1.0, 0.0]])
    y = np.array([0, 1])
    model = KNNClassifier(n_neighbors=2, weights="uniform")
    model.fit(X, y)
    pred = model.predict(np.array([[0.0, 0.0]]))
    # 两边各一个，平局时应选较小标签 0
    assert int(pred[0]) == 0


def test_distance_weights_zero_distance_duplicate():
    # 测试点与训练样本完全重合；distance 权重应强烈偏向该点标签
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 0, 0])  # 与 [0,0] 重合的点标签=1
    model = KNNClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)
    pred = model.predict(np.array([[0.0, 0.0]]))
    assert int(pred[0]) == 1
