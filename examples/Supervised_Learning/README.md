# examples/Supervised_Learning/README.md

This folder contains **end-to-end notebooks** for supervised learning algorithms implemented in `rice_ml.supervised_learning`.

Each notebook is structured as a full workflow rather than a minimal API demo. In general you will see:

- dataset loading and quick exploration,
- preprocessing (standardization when needed, train/test split),
- model training and hyperparameter experiments,
- evaluation (accuracy, confusion matrix, RMSE/R² when relevant),
- visualizations (decision boundaries, loss curves, performance vs hyperparameters),
- a short conclusion summarizing results and limitations.

## Subfolders

- `K_Nearest_Neighbors/`  
  Distance-based learning for classification/regression; shows how **k** affects bias/variance.

- `Linear_Regression/`  
  Baseline regression model; emphasizes metrics such as **RMSE** and **R²**.

- `Logistic_Regression/`  
  Probabilistic linear classifier; focuses on decision boundaries and classification metrics.

- `Perceptron/`  
  Classic mistake-driven linear classifier; highlights linear separability and convergence behavior.

- `Multilayer_Perceptron/`  
  Neural network classifier; demonstrates non-linear decision boundaries, optimization, and scaling sensitivity.

- `Decision_Trees/`  
  CART-style tree classifier; interpretable splits and depth/leaf-size tradeoffs.

- `Regression_Trees/`  
  Tree-based regression; highlights overfitting and the role of pruning-style hyperparameters.

- `Ensemble_Methods/`  
  Combines multiple learners (e.g., random forest-style bagging) to improve generalization.

## Notes on preprocessing

Some supervised algorithms are **scale-sensitive** (e.g., logistic regression, MLP, often KNN), while tree-based methods are typically not.

In these notebooks, standardization is performed using **training data only**, then applied to the test set using the saved parameters to avoid data leakage.

## Running the notebooks

From the repository root:

```bash
pip install -e .
jupyter notebook
```

Then open any notebook under `examples/Supervised_Learning/`.
