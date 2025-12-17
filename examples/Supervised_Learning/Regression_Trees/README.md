# Regression Trees

This example demonstrates a **CART-style Regression Tree** implemented in `rice_ml`:

- **Model:** `RegressionTreeRegressor` (`src/rice_ml/supervised_learning/regression_trees.py`)
- **Criterion:** weighted variance reduction (equivalently minimizing SSE / MSE)
- **Implementation:** NumPy-only (no scikit-learn models)

## Notebook

`Regression_Trees.ipynb` walks through a complete workflow:

1. **Set-up**: import from `src/rice_ml` (adds `<repo>/src` to `sys.path`)
2. **Helpers**: RMSE and R² (NumPy-only)
3. **Part A — Diabetes regression**
   - train/test split
   - (optional) standardization fitted on train only
   - baseline tree + hyperparameter sweeps:
     - `max_depth` (bias–variance tradeoff)
     - `min_samples_leaf` (regularization)
     - `max_features` (feature subsampling)
4. **Part B — 1D synthetic regression**
   - visualizes the **piecewise-constant** predictions of regression trees
   - shows how depth affects fit and generalization

## How to run

From the repo root, either:

- **Option A (recommended):**
  ```bash
  pip install -e .
  ```
  then open and run the notebook.

- **Option B:**
  Open the notebook inside the repo; it automatically searches upward for `src/rice_ml`
  and adds it to `sys.path`.

## Notes

- Trees typically do **not** require feature scaling, but the notebook keeps the same
  preprocessing pattern used in other examples (fit transforms on **train only** and
  apply to **test**) to avoid data leakage and keep a consistent pipeline.
