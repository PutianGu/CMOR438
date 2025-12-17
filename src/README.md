# src/README.md

This folder contains the **core Python package** for this project: `rice_ml`.

The goal of `rice_ml` is to provide clean implementations of common machine learning algorithms, along with lightweight utilities for preprocessing and evaluation. All public APIs are designed to be readable and consistent across modules.

## Package layout

```text
src/
└── rice_ml/
    ├── __init__.py
    ├── processing/
    │   ├── preprocessing.py      # scaling / standardization helpers (train-only stats)
    │   ├── post_processing.py    # evaluation metrics (accuracy, RMSE, confusion matrix, etc.)
    │   └── __init__.py
    ├── supervised_learning/
    │   ├── distance_metrics.py   # reusable distance functions for KNN
    │   ├── k_nearest_neighbors.py
    │   ├── linear_regression.py
    │   ├── logistic_regression.py
    │   ├── perceptron.py
    │   ├── multilayer_perceptron.py
    │   ├── decision_trees.py
    │   ├── regression_trees.py
    │   ├── ensemble_methods.py
    │   └── __init__.py
    └── unsupervised_learning/
        ├── k_means_clustering.py
        ├── pca.py
        ├── dbscan.py
        ├── community_detection.py
        └── __init__.py
```

## Design principles

- **Teaching-first implementations:** prioritize clarity and correctness over micro-optimizations.
- **NumPy-based:** core learning algorithms avoid scikit-learn for the model logic.
- **Consistent interfaces:** most estimators follow `fit(...)` → `predict(...)` and optionally `predict_proba(...)`.
- **Safe preprocessing:** any standardization in notebooks is done using **train-only** statistics, then applied to test data to avoid leakage.

## Typical usage

```python
import numpy as np
from rice_ml.supervised_learning import DecisionTreeClassifier
from rice_ml.processing.preprocessing import standardize

X = np.random.randn(100, 2)
y = (X[:, 0] > 0).astype(int)

X_std, params = standardize(X, return_params=True)

clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_std, y)
pred = clf.predict(X_std)
```

> For complete demonstrations (EDA, preprocessing, evaluation, plots), see the notebooks in `examples/`.
