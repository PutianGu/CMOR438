# examples/Unsupervised_Learning/README.md

This folder contains **end-to-end notebooks** for unsupervised learning algorithms implemented in `rice_ml.unsupervised_learning`.

Unlike supervised learning, unsupervised workflows emphasize:
- exploratory analysis and visualization,
- choosing hyperparameters without labels,
- interpreting clusters/components,
- sanity checks and qualitative evaluation.

## Subfolders

- `K_Means_Clustering/`  
  Partition-based clustering that minimizes within-cluster variance; shows sensitivity to **K** and initialization.

- `PCA/`  
  Dimensionality reduction via orthogonal components maximizing variance; focuses on explained variance and projection.

- `DBSCAN/`  
  Density-based clustering that can find non-spherical clusters and identify outliers; emphasizes `eps` and `min_samples`.

- `Community_Detection/`  
  Graph-based clustering / label propagation-style community discovery; focuses on structure emerging from connectivity.

## Common notebook pattern

Most notebooks include:
- synthetic or real dataset loading,
- preprocessing when needed (e.g., scaling for distance-based methods),
- running the algorithm across key hyperparameters,
- plots to interpret results (cluster assignments, component projections, etc.),
- a conclusion describing what worked, what didn’t, and why.

## Running the notebooks

From the repository root:

```bash
pip install -e .
jupyter notebook
```

Then open any notebook under `examples/Unsupervised_Learning/`.
