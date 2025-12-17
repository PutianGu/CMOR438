# Perceptron

This directory contains example code and notes for the **Perceptron** algorithm in supervised learning.

## Algorithm

The Perceptron is a **linear classifier** trained with a simple mistake-driven update rule.

### Model

For input features `x`:

- Score for class `k`: `s_k = w_k · x + b_k`
- Prediction: `argmax_k s_k`

(Binary classification is a special case of the same idea.)

### Training rule (multiclass)

When a training example `(x, y_true)` is misclassified as `y_pred`, update:

w_true = w_true + η * x
w_pred = w_pred - η * x

where:
- `η` is the learning rate
- `w_true` is the weight vector for the correct class
- `w_pred` is the weight vector for the predicted (wrong) class

This pushes the decision boundary so that the correct class scores higher next time.

### Practical notes

- The Perceptron converges only when classes are **linearly separable**.
- Feature scaling (e.g., standardization) is often helpful.

## Data (used in the example notebook)

The Perceptron example notebook will use **scikit-learn datasets**:
- Binary classification: `Breast Cancer Wisconsin` dataset
- Multiclass classification: `Iris` dataset

The notebook will standardize features, do a train/test split, and report accuracy + learning curves.

