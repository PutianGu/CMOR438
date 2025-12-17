# Multilayer Perceptron (MLP)

This folder demonstrates a **NumPy-only** neural network classifier implemented in `rice_ml`:

- `MultilayerPerceptronClassifier` (alias: `MLPClassifier`)
- Module: `src/rice_ml/supervised_learning/multilayer_perceptron.py`

## What this notebook covers

`Multilayer_Perceptron.ipynb` walks through:

1. **XOR (toy example)**
   - Shows why hidden layers + nonlinear activations are necessary for non-linearly separable data.

2. **Iris (2D) multiclass classification**
   - Train/test split + **standardization**
   - Decision-region visualization
   - Small hyperparameter sweeps (hidden size, learning rate)

3. **Breast Cancer (30D) binary classification**
   - Training with **Adam** vs **SGD**
   - Loss-curve visualization and generalization check

## Key takeaways

- MLPs learn nonlinear boundaries via **hidden layers** and **activations** (ReLU/tanh).
- **Feature scaling matters** (unlike decision trees): standardization usually improves convergence.
- Optimizer choice matters: **Adam** tends to be more stable than SGD in small educational setups.
- Regularization can be achieved via `alpha` (L2 weight decay) and `early_stopping`.

## How to run

From the repository root:

```bash
pip install -e .
jupyter notebook
```

Open:

- `examples/Supervised_Learning/Multilayer_Perceptron/Multilayer_Perceptron.ipynb`

## Notes

- We use scikit-learn **datasets only** (no scikit-learn models).
- The implementation is intentionally compact and educational rather than highly optimized.
