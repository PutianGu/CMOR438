import numpy as np
import pytest
from rice_ml.supervised_learning.preprocessing import StandardScaler


def test_standard_scaler_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    ss = StandardScaler().fit(X)
    Z = ss.transform(X)
    X_inv = ss.inverse_transform(Z)

    assert np.allclose(X, X_inv)
    assert np.allclose(Z.mean(0), 0, atol=1e-7)


def test_transform_before_fit_raises():
    ss = StandardScaler()
    with pytest.raises(RuntimeError):
        ss.transform([[1.0, 2.0]])


def test_inverse_before_fit_raises():
    ss = StandardScaler()
    with pytest.raises(RuntimeError):
        ss.inverse_transform([[0.0, 0.0]])
