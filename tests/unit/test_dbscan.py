import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN


def _partition_signature(labels: np.ndarray):
    """
    Convert labels into a canonical partition signature:
    - noise_count
    - sorted list of sorted index-tuples for each cluster (ignoring label IDs)
    """
    labels = np.asarray(labels)
    noise = int(np.sum(labels == -1))
    clusters = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = tuple(sorted(np.flatnonzero(labels == lab).tolist()))
        clusters.append(idx)
    clusters = sorted(clusters)  # canonical order
    return noise, clusters


def test_dbscan_two_clusters_plus_noise():
    # two tight clusters + one far noise
    X = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
        [50.0, 50.0],
    ], dtype=float)

    db = DBSCAN(eps=0.25, min_samples=2)
    labels = db.fit_predict(X)

    assert labels.shape == (6,)
    assert db.n_clusters_ == 2
    assert np.sum(labels == -1) == 1  # the far point
    # cluster membership sizes should be [2, 3] (order irrelevant)
    sizes = sorted([np.sum(labels == k) for k in range(db.n_clusters_)])
    assert sizes == [2, 3]
    assert db.core_sample_indices_.ndim == 1


def test_dbscan_all_noise_when_eps_tiny():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [2.0, 2.0]], dtype=float)
    labels = DBSCAN(eps=1e-6, min_samples=2).fit_predict(X)
    assert np.all(labels == -1)


def test_dbscan_single_cluster_when_eps_large():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [2.0, 2.0]], dtype=float)
    db = DBSCAN(eps=10.0, min_samples=1).fit(X)
    labels = db.labels_
    assert db.n_clusters_ == 1
    assert np.all(labels == 0)


def test_dbscan_input_validation():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0]], dtype=float)

    with pytest.raises(ValueError):
        DBSCAN(eps=0.0, min_samples=2).fit(X)

    with pytest.raises(ValueError):
        DBSCAN(eps=-1.0, min_samples=2).fit(X)

    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=0).fit(X)

    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=2).fit([[0.0, np.nan], [1.0, 1.0]])


def test_dbscan_invariance_to_permutation_up_to_label_ids():
    X = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
        [50.0, 50.0],
    ], dtype=float)

    db1 = DBSCAN(eps=0.25, min_samples=2)
    labels1 = db1.fit_predict(X)
    sig1 = _partition_signature(labels1)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    Xp = X[perm]

    db2 = DBSCAN(eps=0.25, min_samples=2)
    labels2p = db2.fit_predict(Xp)

    # map permuted labels back to original order
    labels2 = np.empty_like(labels2p)
    labels2[perm] = labels2p

    sig2 = _partition_signature(labels2)
    assert sig1 == sig2
