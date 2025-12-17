# CMOR 438 — Machine Learning Algorithms (Rice University)

A compact, NumPy-first machine learning library built for **CMOR 438: Machine Learning Algorithms** at Rice University.  
This repository implements classic supervised and unsupervised algorithms **from scratch**, with **pytest unit tests** and **end-to-end Jupyter notebooks** demonstrating complete ML workflows (data → preprocessing → training → evaluation → visualization).

- **Author:** Putian Gu  
- **Email:** putian.gu@rice.edu  
- **Repository:** https://github.com/PutianGu/CMOR438  

---

## Highlights

- **From-scratch implementations (NumPy):** models are implemented without relying on scikit-learn for the core logic.
- **Consistent, readable APIs:** `fit`, `predict`, `predict_proba` (when applicable).
- **Preprocessing utilities included:** standardization and evaluation helpers live in `rice_ml/processing/`.
- **Reproducible experiments:** `random_state` supported where randomness is involved.
- **Teaching-first notebooks:** each notebook walks through *why* each step matters, not just “how to call the function.”

---

## Implemented algorithms

### Supervised learning
- K-Nearest Neighbors (Classifier & Regressor)
- Linear Regression
- Logistic Regression
- Perceptron
- Multilayer Perceptron (Neural Network)
- Decision Trees (Classifier)
- Regression Trees (Regressor)
- Ensemble Methods (e.g., Random Forest-style bagging utilities)

### Unsupervised learning
- K-Means Clustering
- PCA (Principal Component Analysis)
- DBSCAN
- Community Detection (graph / label propagation style)

### Utilities
- Preprocessing (e.g., standardization)
- Post-processing metrics (e.g., accuracy, RMSE, confusion matrix)

---

## Installation

Clone and install in editable mode:

```bash
git clone https://github.com/PutianGu/CMOR438.git
cd CMOR438
pip install -e .
```

---

## Quick start

```python
import numpy as np

from rice_ml.supervised_learning import LogisticRegression
from rice_ml.processing.preprocessing import standardize
from rice_ml.processing.post_processing import accuracy_score

X = np.random.randn(200, 5)
y = (X[:, 0] - 0.25 * X[:, 1] > 0).astype(int)

# Fit preprocessing on TRAIN only in real workflows
X_std, _params = standardize(X, return_params=True)

clf = LogisticRegression(max_iter=2000, lr=0.1).fit(X_std, y)
pred = clf.predict(X_std)

print("train accuracy:", accuracy_score(y, pred))
```

---

## Example notebooks

Notebooks live under `examples/` and are designed as full pipelines:
- load and inspect a dataset,
- preprocess features (train-only stats),
- train a model and explore key hyperparameters,
- evaluate with quantitative metrics,
- visualize results (decision regions, loss curves, performance plots).

---

## Repository structure

A “slightly deeper” map of the repo (major folders only):

```text
CMOR438/
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       └── tests.yml
├── examples/
│   ├── README.md
│   ├── Supervised_Learning
│   │   ├── README.md
│   │   ├── K_Nearest_Neighbors        # Non-parametric, distance-based model for classification/regression.
│   │   ├── Linear_Regression          # Predicts continuous targets via least-squares linear fit.
│   │   ├── Logistic_Regression        # Linear classifier using a sigmoid/softmax to model class probabilities.
│   │   ├── Perceptron                 # Classic linear classifier trained with mistake-driven updates.
│   │   ├── Multilayer_Perceptron      # Feedforward neural network for non-linear decision boundaries.
│   │   ├── Decision_Trees             # CART-style classifier using recursive splits (e.g., Gini impurity).
│   │   ├── Regression_Trees           # Tree-based regression using recursive splits to minimize squared error.
│   │   └── Ensemble_Methods           # Combines many weak learners (e.g., random forests) for better generalization.
│   └── Unsupervised_Learning
│       ├── README.md
│       ├── K_Means_Clustering         # Partitions data into K clusters by minimizing within-cluster variance.
│       ├── PCA                        # Dimensionality reduction via orthogonal components maximizing variance.
│       ├── DBSCAN                     # Density-based clustering that finds arbitrary-shape clusters + outliers.
│       └── Community_Detection        # Graph clustering to discover communities (densely connected node groups).
├── src/
│   ├── README.md
│   └── rice_ml/
│       ├── __init__.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   └── post_processing.py
│       ├── supervised_learning/
│       │   ├── __init__.py
│       │   ├── distance_metrics.py
│       │   ├── k_nearest_neighbors.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── perceptron.py
│       │   ├── multilayer_perceptron.py
│       │   ├── decision_trees.py
│       │   ├── regression_trees.py
│       │   └── ensemble_methods.py
│       └── unsupervised_learning/
│           ├── __init__.py
│           ├── k_means_clustering.py
│           ├── pca.py
│           ├── dbscan.py
│           └── community_detection.py
├── tests/
│   └── unit/
│       ├── test_preprocessing.py
│       ├── test_post_processing.py
│       ├── test_distances.py
│       ├── test_knn.py
│       ├── test_linear_regression.py
│       ├── test_logistic_regression.py
│       ├── test_perceptron.py
│       ├── test_decision_trees.py
│       ├── test_regression_trees.py
│       ├── test_ensemble_methods.py
│       ├── test_kmeans.py
│       ├── test_pca.py
│       ├── test_dbscan.py
│       └── test_community_detection.py
├── pyproject.toml
├── LICENSE
└── README.md
```

> Note: local folders like `.venv/` and `.pytest_cache/` are intentionally omitted from the structure above (they are environment artifacts, not part of the library).

---

## Testing

Run all unit tests:

```bash
pytest
```

Run a specific module:

```bash
pytest tests/unit/test_decision_trees.py
```

CI runs tests via `.github/workflows/tests.yml`.

---

## License

This project is released under the **MIT License**. See `LICENSE`.

## AI Disclaimer

Generative AI (ChatGPT 5.2 and Gemini) were used to assist in developing repo for this project, primarily for help with algorithm coding.