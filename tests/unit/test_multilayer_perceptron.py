import numpy as np
import pytest

from rice_ml.supervised_learning.multilayer_perceptron import MultilayerPerceptronClassifier


def test_predict_before_fit_raises():
    clf = MultilayerPerceptronClassifier()
    X = np.array([[0.0, 0.0]])
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)


def test_predict_proba_shape_and_row_sums_binary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    clf = MultilayerPerceptronClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        learning_rate=0.01,
        max_iter=200,
        batch_size=32,
        random_state=0,
    ).fit(X, y)

    P = clf.predict_proba(X)
    assert P.shape == (X.shape[0], 2)
    assert np.all(P >= 0.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_reproducible_with_random_state():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(250, 4))
    y = (X[:, 0] - 0.2 * X[:, 2] > 0).astype(int)

    clf1 = MultilayerPerceptronClassifier(
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="adam",
        learning_rate=0.01,
        max_iter=150,
        batch_size=32,
        random_state=7,
    ).fit(X, y)

    clf2 = MultilayerPerceptronClassifier(
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="adam",
        learning_rate=0.01,
        max_iter=150,
        batch_size=32,
        random_state=7,
    ).fit(X, y)

    assert np.array_equal(clf1.predict(X), clf2.predict(X))


def test_xor_learnable_with_hidden_layer_tanh_adam():
    # XOR requires non-linear decision boundary
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    clf = MultilayerPerceptronClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        solver="adam",
        learning_rate=0.05,
        max_iter=800,
        batch_size=4,
        random_state=0,
        tol=1e-8,
        n_iter_no_change=30,
        early_stopping=False,
    ).fit(X, y)

    pred = clf.predict(X)
    assert np.array_equal(pred, y)


def test_invalid_hyperparams_raise():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier(hidden_layer_sizes=()).fit(X, y)

    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier(hidden_layer_sizes=(0,)).fit(X, y)

    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier(activation="sigmoid").fit(X, y)

    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier(solver="rmsprop").fit(X, y)

    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier(learning_rate=0.0).fit(X, y)


def test_invalid_shapes_raise():
    clf = MultilayerPerceptronClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))  # X not 2D
    with pytest.raises(ValueError):
        clf.fit(np.array([[1.0, 2.0]]), np.array([[0]]))  # y not 1D
    with pytest.raises(ValueError):
        clf.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0]))  # mismatch


def test_nan_raises():
    X = np.array([[0.0, np.nan], [1.0, 1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        MultilayerPerceptronClassifier().fit(X, y)
