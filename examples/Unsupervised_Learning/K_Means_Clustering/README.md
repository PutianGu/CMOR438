# K-Means Clustering

## Overview
K-Means partitions data into `k` compact clusters by minimizing the within-cluster sum of squared distances to cluster centers. It works best when clusters are roughly spherical and well-separated in feature space.

## Algorithm
- Initialize `k` cluster centers (e.g., randomly).
- Repeat until convergence:
  - Assign each point to the nearest center.
  - Update each center as the mean of points assigned to it.
- Convergence is typically determined by small center movement (`tol`) or reaching `max_iters`.

## Key Parameters
- `k`: Number of clusters.
- `max_iters`: Maximum iterations.
- `tol`: Convergence tolerance.
- `random_state` (if supported): seed for reproducibility.

## Complexity
- Dominated by distance computations: `O(n * k * t)` where `t` is the number of iterations.

## Strengths & Trade-offs
- Pros:
  - Simple and fast in practice.
  - Scales well for moderate to large datasets.
- Cons:
  - Requires choosing `k`.
  - Assumes spherical/compact clusters.
  - Sensitive to initialization and feature scaling.

## Data
- Dataset: Iris (150 samples, 4 numeric features).
- Features: sepal length/width, petal length/width (cm).
- Labels: available but not used during training; useful for visualization/qualitative comparison.
- Preprocessing: feature-wise Z-score standardization is recommended for distance-based clustering.
- Notebook parameters (typical): `k=3`, `max_iters=100`, `tol=1e-4`, `random_state=42`.

