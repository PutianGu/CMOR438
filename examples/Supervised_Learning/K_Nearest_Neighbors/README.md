# K-Nearest Neighbors (KNN)

This directory contains an example workflow for **K-Nearest Neighbors (KNN)** in supervised learning,
implemented with the NumPy-only models in `rice_ml`.

## Algorithm

KNN is a **non-parametric** method:
- For **classification**, it predicts the label by looking at the most common class among the `k` nearest training samples.
- For **regression**, it predicts a value by averaging the targets of the `k` nearest training samples.

Key hyperparameters:
- `n_neighbors (k)`: how many neighbors to use.
- `metric`: distance metric (`euclidean` or `manhattan`).
- `weights`:
  - `uniform`: each neighbor contributes equally
  - `distance`: closer neighbors contribute more (inverse-distance weighting)

## Data used in the notebook

To keep examples realistic and reproducible, we use datasets from `sklearn.datasets`:

- **Classification**: Iris dataset (3 classes, 4 numeric features).
- **Regression**: Diabetes dataset (numeric target, 10 numeric features).

Typical preprocessing:
- split into train/test
- standardize features using `rice_ml.processing.preprocessing.standardize`
  (important for distance-based methods)

## What you should see

- Classification: accuracy on the Iris test set should be high with a reasonable `k`
- Regression: R^2 should be comparable to a simple baseline, and performance will depend on `k` and scaling

