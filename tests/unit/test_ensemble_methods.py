import numpy as np
import pytest

from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier, RandomForestRegressor


def test_rf_classifier_predict_proba_shape_and_sums():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # contiguous {0,1}

    clf = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=42).fit(X, y)
    P = clf.predict_proba(X)

    assert P.shape == (X.shape[0], 2)
    assert np.all(P >= 0.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_rf_classifier_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] - 0.2 * X[:, 2] > 0).astype(int)

    clf1 = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=7).fit(X, y)
    clf2 = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=7).fit(X, y)

    assert np.array_equal(clf1.predict(X), clf2.predict(X))


def test_rf_classifier_invalid_n_estimators_raises():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0).fit(np.ones((2, 1)), np.array([0, 1], dtype=int))


def test_rf_regressor_improves_over_mean_baseline():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(400, 3))
    y = (X[:, 0] > 0).astype(float) + 0.2 * X[:, 1] + rng.normal(0.0, 0.05, size=400)

    baseline = np.full_like(y, y.mean())
    baseline_mse = np.mean((y - baseline) ** 2)

    model = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=0).fit(X, y)
    pred = model.predict(X)
    mse = np.mean((y - pred) ** 2)

    assert mse < baseline_mse


def test_rf_regressor_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(250, 6))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=250)

    m1 = RandomForestRegressor(n_estimators=15, max_depth=5, random_state=123).fit(X, y)
    m2 = RandomForestRegressor(n_estimators=15, max_depth=5, random_state=123).fit(X, y)

    assert np.allclose(m1.predict(X), m2.predict(X), atol=1e-12)
