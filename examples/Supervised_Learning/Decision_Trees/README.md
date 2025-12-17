# Decision Trees (CART) — Supervised Learning

This folder contains an example notebook demonstrating a **CART-style Decision Tree Classifier** implemented from scratch in the `rice_ml` package (NumPy-only).

## Files

- `Decision_Trees.ipynb` — end-to-end workflow:
  - dataset loading (Iris + Breast Cancer)
  - train/test split
  - preprocessing (standardization shown, but discussed as optional for trees)
  - model fitting + evaluation
  - hyperparameter sweeps (`max_depth`, `min_samples_leaf`, `max_features`)
  - 2D decision region visualizations (Iris)

## Algorithm overview

The classifier follows the classic CART recipe:

1. Start at the root with all training data.
2. For each candidate split `(feature, threshold)`, compute the **Gini impurity** of the left and right partitions:
   - Gini(y) = 1 - sum_k p_k^2
3. Choose the split that minimizes the weighted impurity.
4. Recurse on left and right child nodes until a stopping criterion is met.

### Stopping / regularization controls

- `max_depth`: limits tree depth (controls complexity)
- `min_samples_split`: minimum samples needed to attempt a split
- `min_samples_leaf`: minimum samples allowed in any leaf
- `max_features`: optional feature subsampling (useful later for ensembles)

## Data + preprocessing notes

The notebook uses:
- **Iris** for a 2D multiclass visualization
- **Breast Cancer Wisconsin** for a realistic, higher-dimensional binary classification task

### Why standardization is optional for decision trees
Decision trees split by comparing values to thresholds. Because scaling is a monotone transform (with positive scale), the *ordering* of samples within each feature is preserved, so trees typically do **not** require feature scaling.  

That said, the notebook still shows a standardization step to match the course workflow across models and to make comparisons consistent.


## Expected output

You should see:
- accuracy and confusion matrices for both datasets
- decision region plots (Iris 2D)
- train vs test accuracy curves illustrating overfitting as depth increases
