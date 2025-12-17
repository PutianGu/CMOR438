import numpy as np
import pytest

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier


def test_basic_fit_predict_perfect_on_simple_split():
    # perfectly separable by feature 0
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.array([0, 0, 1, 1], dtype=int)

    clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    pred = clf.predict(X)

    assert pred.shape == y.shape
    assert np.array_equal(pred, y)


def test_predict_proba_shape_and_row_sums():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    # 3-class toy labels, already contiguous 0..2
    y = np.argmax(np.c_[X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]], axis=1).astype(int)

    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    P = clf.predict_proba(X)

    assert P.shape == (X.shape[0], 3)
    assert np.all(P >= 0.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_predict_before_fit_raises():
    X = np.array([[0., 0.], [1., 1.]])
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)


def test_invalid_y_type_and_encoding_raise():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

    # non-integer labels
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, np.array([0., 0., 1., 1.]))

    # integer but not starting at 0
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, np.array([1, 1, 2, 2], dtype=int))

    # integer but not contiguous 0..K-1
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, np.array([0, 2, 2, 0], dtype=int))


def test_invalid_X_shapes_raise():
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([1., 2., 3.]), np.array([0, 1, 0], dtype=int))  # X not 2D
    with pytest.raises(ValueError):
        clf.fit(np.array([[1., 2.]]), np.array([[0]], dtype=int))  # y not 1D


def test_max_features_invalid_raises():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.array([0, 0, 1, 1], dtype=int)

    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_features=0).fit(X, y)

    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_features=3).fit(X, y)  # n_features=2

    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_features=1.5).fit(X, y)  # float must be (0,1]


def test_random_state_reproducible_with_feature_subsampling():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)

    clf1 = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=7).fit(X, y)
    clf2 = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=7).fit(X, y)

    assert np.array_equal(clf1.predict(X), clf2.predict(X))


def test_large_min_samples_leaf_prevents_splitting():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.array([0, 0, 1, 1], dtype=int)

    clf = DecisionTreeClassifier(min_samples_leaf=3, random_state=0).fit(X, y)
    pred = clf.predict(X)

    # with no split, all predictions should be the majority class (tie -> argmax -> class 0)
    assert np.all(pred == 0)
