import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import PerceptronClassifier


def _make_binary_separable(n=120, seed=0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    X0 = rng.normal(loc=(-2.0, -2.0), scale=0.4, size=(n0, 2))
    X1 = rng.normal(loc=(2.0, 2.0), scale=0.4, size=(n1, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def _make_multiclass_separable(n=180, seed=1):
    rng = np.random.default_rng(seed)
    n0 = n // 3
    n1 = n // 3
    n2 = n - n0 - n1
    X0 = rng.normal(loc=(-3.0, 0.0), scale=0.35, size=(n0, 2))
    X1 = rng.normal(loc=(3.0, 0.0), scale=0.35, size=(n1, 2))
    X2 = rng.normal(loc=(0.0, 3.0), scale=0.35, size=(n2, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array(["A"] * n0 + ["B"] * n1 + ["C"] * n2, dtype=object)
    return X, y


def test_predict_before_fit_raises():
    clf = PerceptronClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[0.0, 0.0]])


def test_binary_separable_converges_high_accuracy():
    X, y = _make_binary_separable(n=120, seed=0)
    clf = PerceptronClassifier(max_iter=100, learning_rate=1.0, shuffle=True, random_state=0).fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.98
    assert clf.n_iter_ is not None and clf.n_iter_ <= 100
    assert clf.coef_ is not None and clf.intercept_ is not None


def test_multiclass_separable_high_accuracy():
    X, y = _make_multiclass_separable(n=180, seed=1)
    clf = PerceptronClassifier(max_iter=200, learning_rate=1.0, shuffle=True, random_state=1).fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.95
    # multiclass shapes
    assert clf.coef_ is not None and clf.coef_.shape == (3, 2)
    assert clf.intercept_ is not None and clf.intercept_.shape == (3,)


def test_partial_fit_initializes_and_improves():
    X, y = _make_binary_separable(n=120, seed=2)
    # split into two batches
    X1, y1 = X[:60], y[:60]
    X2, y2 = X[60:], y[60:]

    clf = PerceptronClassifier(max_iter=1, shuffle=False, random_state=0)
    clf.partial_fit(X1, y1, classes=[0, 1])
    acc1 = clf.score(X, y)
    clf.partial_fit(X2, y2)
    acc2 = clf.score(X, y)

    assert acc2 >= acc1
    assert clf.n_iter_ == 2


def test_input_validation_errors():
    clf = PerceptronClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))  # X not 2D
    with pytest.raises(ValueError):
        clf.fit(np.zeros((3, 2)), np.zeros((3, 1)))  # y not 1D
    with pytest.raises(ValueError):
        clf.fit(np.zeros((3, 2)), np.array([0, 1]))  # length mismatch
    with pytest.raises(ValueError):
        PerceptronClassifier(max_iter=0).fit(np.zeros((3, 2)), np.array([0, 1, 0]))


def test_deterministic_with_random_state():
    X, y = _make_binary_separable(n=120, seed=3)
    clf1 = PerceptronClassifier(max_iter=50, shuffle=True, random_state=123).fit(X, y)
    clf2 = PerceptronClassifier(max_iter=50, shuffle=True, random_state=123).fit(X, y)
    assert np.allclose(clf1.coef_, clf2.coef_)
    assert np.allclose(clf1.intercept_, clf2.intercept_)
