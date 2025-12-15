# PCA (Principal Component Analysis)

## Overview
PCA is a linear dimensionality reduction technique that projects data onto a lower-dimensional subspace while preserving as much variance as possible. The new axes (principal components) are orthogonal directions ranked by explained variance.

## Algorithm
- Center the data (and often standardize features).
- Compute principal components via eigen-decomposition of the covariance matrix or via SVD.
- Project data onto the top `n_components` components.

## Key Parameters
- `n_components`: Number of principal components to keep.
- (Optional) `whiten`: rescales components to unit variance (if implemented).

## Complexity
- Depends on method and shapes; commonly around `O(n * d^2)` for covariance-based approaches or SVD variants, where `n` is samples and `d` is features.

## Strengths & Trade-offs
- Pros:
  - Reduces dimensionality and noise.
  - Helps visualization (2D/3D) and can improve downstream learning stability.
- Cons:
  - Linear method (cannot capture nonlinear structure).
  - Components can be harder to interpret in original feature space.

## Data
- Dataset: Iris (150 samples, 4 numeric features).
- Preprocessing: standardization is commonly applied so each feature contributes comparably to variance.
- Usage in notebook: project to 2 components for visualization; labels can be used only for plotting/interpretation.

