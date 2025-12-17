import numpy as np
import pytest

from rice_ml.supervised_learning.regression_trees import RegressionTreeRegressor


def test_fit_predict_shapes_and_dtype():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = RegressionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (X.shape[0],)
    assert np.issubdtype(pred.dtype, np.floating)


def test_piecewise_constant_can_fit_perfectly():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = RegressionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    pred = model.predict(X)

    assert np.allclose(pred, y, atol=1e-12)


def test_mse_improves_vs_mean_baseline_on_nonlinear_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(300, 2))
    # nonlinear-ish target with piecewise structure + noise
    y = (X[:, 0] > 0).astype(float) + 0.5 * (X[:, 1] > 0).astype(float) + rng.normal(0.0, 0.05, size=300)

    # baseline predictor = mean
    baseline = np.full_like(y, y.mean())
    baseline_mse = np.mean((y - baseline) ** 2)

    model = RegressionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    pred = model.predict(X)
    tree_mse = np.mean((y - pred) ** 2)

    assert tree_mse < baseline_mse


def test_predict_before_fit_raises():
    X = np.array([[0.0], [1.0]])
    model = RegressionTreeRegressor()
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_invalid_shapes_raise():
    model = RegressionTreeRegressor()
    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))  # X not 2D
    with pytest.raises(ValueError):
        model.fit(np.array([[1.0, 2.0]]), np.array([[1.0]]))  # y not 1D
    with pytest.raises(ValueError):
        model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0]))  # n mismatch


def test_nan_raises():
    X = np.array([[0.0], [np.nan]])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        RegressionTreeRegressor().fit(X, y)

    X2 = np.array([[0.0], [1.0]])
    y2 = np.array([0.0, np.nan])
    with pytest.raises(ValueError):
        RegressionTreeRegressor().fit(X2, y2)


def test_max_features_invalid_raises():
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        RegressionTreeRegressor(max_features=0).fit(X, y)
    with pytest.raises(ValueError):
        RegressionTreeRegressor(max_features=3).fit(X, y)  # n_features=2
    with pytest.raises(ValueError):
        RegressionTreeRegressor(max_features=1.2).fit(X, y)  # float must be (0,1]


def test_random_state_reproducible_with_feature_subsampling():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(400, 6))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=400)

    m1 = RegressionTreeRegressor(max_depth=4, max_features=3, random_state=7).fit(X, y)
    m2 = RegressionTreeRegressor(max_depth=4, max_features=3, random_state=7).fit(X, y)

    assert np.allclose(m1.predict(X), m2.predict(X), atol=1e-12)


def test_large_min_samples_leaf_prevents_splitting_constant_prediction():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = RegressionTreeRegressor(min_samples_leaf=3, max_depth=5, random_state=0).fit(X, y)
    pred = model.predict(X)

    # no valid split -> predict mean everywhere
    assert np.allclose(pred, y.mean(), atol=1e-12)
