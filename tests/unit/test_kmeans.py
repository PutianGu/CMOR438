import numpy as np
import pytest

from rice_ml.unsupervised_learning.k_means_clustering import KMeans

def _sorted_centers(C):
    # sort by first coordinate for stable comparison
    idx = np.argsort(C[:, 0])
    return C[idx]


def test_kmeans_two_blobs_centers_close():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=0.0, scale=0.2, size=(50, 2))
    X2 = rng.normal(loc=10.0, scale=0.2, size=(50, 2))
    X = np.vstack([X1, X2])

    km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X)
    C = _sorted_centers(km.cluster_centers_)
    assert C.shape == (2, 2)
    assert np.allclose(C[0], np.array([0.0, 0.0]), atol=0.5)
    assert np.allclose(C[1], np.array([10.0, 10.0]), atol=0.5)


def test_kmeans_predict_matches_training_labels_shape():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [9., 9.],
                  [10., 10.]], dtype=float)
    km = KMeans(n_clusters=2, n_init=3, random_state=1).fit(X)
    pred = km.predict(X)
    assert pred.shape == (4,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_kmeans_transform_shape():
    X = np.array([[0., 0.],
                  [1., 0.],
                  [10., 10.]], dtype=float)
    km = KMeans(n_clusters=2, n_init=2, random_state=0).fit(X)
    D = km.transform(X)
    assert D.shape == (3, 2)
    assert np.all(D >= 0)


def test_kmeans_errors():
    X = np.array([[0., 0.],
                  [1., 1.]], dtype=float)

    with pytest.raises(ValueError):
        KMeans(n_clusters=0).fit(X)

    with pytest.raises(ValueError):
        KMeans(n_clusters=3).fit(X)  # more clusters than samples

    km = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError):
        km.predict(X)


def test_kmeans_empty_cluster_handling_does_not_crash():
    # can create empty clusters during updates
    X = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=float)
    km = KMeans(n_clusters=3, n_init=3, random_state=0, max_iter=50).fit(X)
    assert km.cluster_centers_.shape == (3, 1)
    assert np.isfinite(km.inertia_)
