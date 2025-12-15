# DBSCAN

## Overview
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. It discovers clusters of arbitrary shape by grouping points in high-density regions and marks points in low-density regions as noise.

## Algorithm
- For each point, find all neighbors within radius `eps`.
- A point is a **core point** if its neighborhood size is at least `min_samples`.
- A cluster is formed by expanding from core points to include all points that are **density-reachable**.
- Points that are not reachable from any core point are labeled as **noise** (`-1`).

## Key Parameters
- `eps`: Neighborhood radius controlling the density threshold.
- `min_samples`: Minimum number of points (including itself) required to form a dense region.

## Complexity
- With a spatial index (not used here): typically around `O(n log n)`.
- With naive pairwise distances (common in teaching implementations): `O(n^2)`.

## Strengths & Trade-offs
- Pros:
  - Finds arbitrary-shaped clusters.
  - Identifies noise/outliers naturally.
  - Does not require specifying `k`.
- Cons:
  - Sensitive to `eps` and `min_samples`.
  - Can struggle when clusters have varying densities.

## Data
- Dataset: Iris (150 samples, 4 numeric features).
- Features: sepal length/width, petal length/width (cm).
- Labels: available but not used during clustering; can be used for qualitative comparison.
- Preprocessing: feature-wise Z-score standardization to avoid scale dominance.
- Notebook parameters (typical): `eps=0.6`, `min_samples=6` on standardized features.

