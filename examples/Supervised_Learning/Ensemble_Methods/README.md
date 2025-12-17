# Ensemble Methods (Random Forests)

This folder demonstrates **bagging** and **random forests** implemented in `rice_ml` (NumPy-only):

- **Classifier:** `RandomForestClassifier` (`src/rice_ml/supervised_learning/ensemble_methods.py`)
- **Regressor:** `RandomForestRegressor` (`src/rice_ml/supervised_learning/ensemble_methods.py`)
- **Base learners:** `DecisionTreeClassifier` and `RegressionTreeRegressor`

## What this notebook covers

`Ensemble_Methods.ipynb` walks through:

1. **Iris (2D) classification**
   - Train/test workflow + optional standardization (for consistency)
   - Decision-region visualization for a random forest
   - Hyperparameter sweeps: `max_depth`, `n_estimators`, `max_features`

2. **Breast Cancer classification (30D)**
   - Comparison: single decision tree vs random forest
   - Sweeps showing how forests stabilize test performance

3. **Diabetes regression**
   - Metrics: RMSE and R²
   - Comparison: single regression tree vs random forest regressor
   - Regularization sweeps: `max_depth`, `min_samples_leaf`, `max_features`

4. **1D synthetic regression**
   - Visual intuition: averaging many piecewise-constant trees produces smoother predictions

## Key takeaways

- **Bagging reduces variance:** averaging many trees typically generalizes better than a single deep tree.
- Increasing **`n_estimators`** usually stabilizes test performance (diminishing returns after a point).
- **`max_features`** is important because it decorrelates trees—this is a key ingredient in random forests.
- Even though trees do not require scaling, keeping a consistent preprocessing pipeline helps avoid leakage and keeps experiments comparable across models.

## How to run

From the repository root:

```bash
# (optional) create/activate your environment, then:
pip install -e .
jupyter notebook
```

Open:

- `examples/Supervised_Learning/Ensemble_Methods/Ensemble_Methods.ipynb`

## Notes

- We use scikit-learn **datasets only** (no scikit-learn models).
- The implementations are intentionally compact and educational rather than highly optimized.
