import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


def test_binary_predict_proba_shape_and_sums():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=float)
    y = np.array([0, 0, 0, 1])

    clf = LogisticRegression(
        learning_rate=0.2,
        max_iter=3000,
        tol=1e-8,
        random_state=0,
    ).fit(X, y)

    P = clf.predict_proba(X)
    assert P.shape == (4, 2)
    assert np.all(P >= 0) and np.all(P <= 1)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_binary_accuracy_reasonable_on_linearly_separable():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    # linear boundary
    y = (X[:, 0] + 0.8 * X[:, 1] > 0).astype(int)

    clf = LogisticRegression(
        learning_rate=0.1,
        max_iter=4000,
        tol=1e-8,
        random_state=0,
    ).fit(X, y)

    acc = clf.score(X, y)
    assert acc >= 0.90


def test_multiclass_softmax_shapes_and_prob_sums():
    rng = np.random.default_rng(1)
    n_per = 80
    C0 = rng.normal(loc=(-2, 0), scale=0.6, size=(n_per, 2))
    C1 = rng.normal(loc=(2, 0), scale=0.6, size=(n_per, 2))
    C2 = rng.normal(loc=(0, 2), scale=0.6, size=(n_per, 2))
    X = np.vstack([C0, C1, C2])
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    clf = LogisticRegression(
        multi_class="multinomial",
        learning_rate=0.2,
        max_iter=5000,
        tol=1e-8,
        random_state=0,
    ).fit(X, y)

    P = clf.predict_proba(X)
    assert P.shape == (3 * n_per, 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert clf.score(X, y) >= 0.90


def test_regularization_shrinks_coef_norm():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5))
    w_true = np.array([2.0, -1.5, 0.5, 0.0, 1.0])
    logits = X @ w_true
    y = (logits > 0).astype(int)

    clf0 = LogisticRegression(learning_rate=0.1, max_iter=4000, tol=1e-8, l2=0.0, random_state=0).fit(X, y)
    clf1 = LogisticRegression(learning_rate=0.1, max_iter=4000, tol=1e-8, l2=5.0, random_state=0).fit(X, y)

    norm0 = np.linalg.norm(clf0.coef_)
    norm1 = np.linalg.norm(clf1.coef_)
    assert norm1 < norm0


def test_invalid_inputs_raise():
    X = np.array([[1., 2.], [3., 4.]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0.0).fit(X, y)

    with pytest.raises(ValueError):
        LogisticRegression(max_iter=0).fit(X, y)

    with pytest.raises(ValueError):
        LogisticRegression(l2=-1.0).fit(X, y)

    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.array([1, 1]))  # only one class

    with pytest.raises(ValueError):
        LogisticRegression().fit(np.array([1., 2., 3.]), y)  # X not 2D
