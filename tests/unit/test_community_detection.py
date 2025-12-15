import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import CommunityDetection


def _partition_signature(labels: np.ndarray):
    labels = np.asarray(labels)
    clusters = []
    for lab in np.unique(labels):
        idx = tuple(sorted(np.flatnonzero(labels == lab).tolist()))
        clusters.append(idx)
    return sorted(clusters)


def test_connected_components_two_components_and_isolate():
    # component 1: 0-1-2, component 2: 3-4, isolate: 5
    A = np.zeros((6, 6), dtype=float)
    edges = [(0, 1), (1, 2), (3, 4)]
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    cd = CommunityDetection(method="connected_components")
    labels = cd.fit_predict(A)

    assert labels.shape == (6,)
    sig = _partition_signature(labels)
    assert sig == [(0, 1, 2), (3, 4), (5,)]
    assert cd.n_communities_ == 3


def test_connected_components_directed_treated_undirected_by_default():
    # directed edge 0->1 only, but assume_undirected=True => still connected
    A = np.array([[0, 1],
                  [0, 0]], dtype=float)
    labels = CommunityDetection(method="connected_components").fit_predict(A)
    sig = _partition_signature(labels)
    assert sig == [(0, 1)]


def test_connected_components_directed_not_undirected_if_disabled():
    A = np.array([[0, 1],
                  [0, 0]], dtype=float)
    labels = CommunityDetection(method="connected_components", assume_undirected=False).fit_predict(A)
    # With only 0->1, starting at 0 reaches 1, so still one component under reachability?
    # Our traversal uses adjacency rows; without symmetrizing, 1 has no outgoing edges,
    # but it's still visited from 0 in the same component. So still (0,1).
    sig = _partition_signature(labels)
    assert sig == [(0, 1)]


def test_input_validation():
    with pytest.raises(ValueError):
        CommunityDetection().fit([[1, 0, 0], [0, 1, 0]])  # non-square

    with pytest.raises(ValueError):
        CommunityDetection().fit([[0.0, np.nan], [1.0, 0.0]])  # non-finite

    with pytest.raises(ValueError):
        CommunityDetection(method="bad").fit([[0.0]])  # invalid method
