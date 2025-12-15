import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


def test_pca_basic_shapes():
    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]])
    pca = PCA(n_components=1).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (3, 1)


def test_pca_full_reconstruction_no_whiten():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    pca = PCA(n_components=None, whiten=False).fit(X)
    Z = pca.transform(X)
    X_rec = pca.inverse_transform(Z)
    assert X_rec.shape == X.shape
    assert np.allclose(X_rec, X, atol=1e-10)


def test_pca_transform_before_fit_raises():
    X = np.array([[1., 2.],
                  [3., 4.]])
    pca = PCA(n_components=1)
    with pytest.raises(RuntimeError):
        pca.transform(X)


def test_pca_float_n_components_selects_reasonable_k():
    # first feature dominates variance
    X = np.array([[0., 0.],
                  [10., 0.],
                  [20., 0.],
                  [30., 0.]], dtype=float)
    pca = PCA(n_components=0.9).fit(X)
    assert pca.n_components_ == 1


def test_pca_whiten_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    pca = PCA(n_components=2, whiten=True).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (50, 2)
    X_rec = pca.inverse_transform(Z)
    assert X_rec.shape == (50, 3)
