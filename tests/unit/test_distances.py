import numpy as np
import pytest
from rice_ml.supervised_learning.functions import (
    euclidean,
    manhattan,
    pairwise_distance,
)


def test_point_distances():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean(a, b) == 5.0
    assert manhattan(a, b) == 7.0


def test_pairwise():
    X1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    X2 = np.array([[3.0, 4.0], [1.0, 1.0]])
    D_e = pairwise_distance(X1, X2, "euclidean")
    D_m = pairwise_distance(X1, X2, "manhattan")

    assert D_e.shape == (2, 2) and D_m.shape == (2, 2)
    assert np.isclose(D_e[0, 0], 5.0)
    assert D_m[0, 1] == 2.0


def test_invalid_metric():
    X = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        pairwise_distance(X, X, metric="chebyshev")
