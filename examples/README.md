# examples/README.md

This directory contains the **demonstration notebooks** for the `rice_ml` package.

The notebooks are organized by learning type:

- `Supervised_Learning/` — models trained with labeled targets (classification/regression)
- `Unsupervised_Learning/` — structure discovery without labels (clustering / dimensionality reduction / graphs)

Each algorithm folder typically includes:
- a Jupyter notebook (`*.ipynb`) with a complete workflow, and
- a local `README.md` describing the dataset(s), what the notebook demonstrates, and key takeaways.

## How to run

From the repository root:

```bash
pip install -e .
jupyter notebook
```

Then navigate into `examples/` and open any notebook.

## Notes on reproducibility

- Notebooks set `random_state` where applicable to make results reproducible.
- Any feature scaling/standardization is computed using **training data only**, and then applied to the test set to avoid leakage.
- Plots are used to support interpretation (decision boundaries, confusion matrices, RMSE/R² curves, clustering visualizations, etc.).
