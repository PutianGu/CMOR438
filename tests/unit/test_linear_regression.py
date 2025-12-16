import numpy as np
import pytest

from rice_ml.supervised_learning.linear_regression import LinearRegression


def test_ols_recovers_known_coefficients_with_intercept():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w_true = np.array([2.0, -3.0, 0.5])
    b_true = 1.25
    y = X @ w_true + b_true

    model = LinearRegression(fit_intercept=True, solver="normal", l2=0.0)
    model.fit(X, y)

    assert np.allclose(model.coef_, w_true, atol=1e-8)
    assert abs(model.intercept_ - b_true) <= 1e-8

    yhat = model.predict(X)
    assert np.max(np.abs(yhat - y)) <= 1e-8
    assert model.score(X, y) == pytest.approx(1.0)


def test_multi_output_shapes_and_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 2))
    w1 = np.array([1.0, 2.0])
    w2 = np.array([-2.0, 0.5])
    b = np.array([0.2, -1.0])

    y1 = X @ w1 + b[0]
    y2 = X @ w2 + b[1]
    Y = np.column_stack([y1, y2])

    model = LinearRegression(fit_intercept=True, solver="normal")
    model.fit(X, Y)

    assert model.coef_.shape == (2, 2)
    assert model.intercept_.shape == (2,)

    Yhat = model.predict(X)
    assert Yhat.shape == (150, 2)
    assert model.score(X, Y) == pytest.approx(1.0)


def test_ridge_handles_collinearity_without_crashing():
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=(100, 1))
    x2 = x1.copy()  # perfectly collinear
    X = np.concatenate([x1, x2], axis=1)
    y = 3.0 * x1[:, 0] + 1.0

    model = LinearRegression(fit_intercept=True, solver="normal", l2=1.0)
    model.fit(X, y)

    # should produce finite parameters and decent fit
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.score(X, y) > 0.9


def test_gd_solver_is_close_to_normal_solution_on_well_conditioned_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    w_true = np.array([1.5, -2.0, 0.7, 0.3])
    b_true = -0.8
    y = X @ w_true + b_true

    ols = LinearRegression(fit_intercept=True, solver="normal")
    ols.fit(X, y)

    gd = LinearRegression(fit_intercept=True, solver="gd", lr=0.2, max_iter=5000, tol=1e-12)
    gd.fit(X, y)

    assert np.allclose(gd.coef_, ols.coef_, atol=1e-6)
    assert abs(gd.intercept_ - ols.intercept_) <= 1e-6
    assert gd.n_iter_ <= 5000
    assert gd.loss_history_ is not None
    assert gd.loss_history_.ndim == 1


def test_input_validation_errors():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        LinearRegression().fit(X.reshape(4), y)  # X not 2D

    with pytest.raises(ValueError):
        LinearRegression().fit(X, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # mismatch

    with pytest.raises(ValueError):
        LinearRegression().fit(np.array([]).reshape(0, 2), y)  # empty X

    with pytest.raises(TypeError):
        LinearRegression().fit([["a", "b"], ["c", "d"]], y)  # non-numeric X
