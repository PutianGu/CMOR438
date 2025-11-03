import numpy as np
from rice_ml.supervised_learning.post_processing import confusion_matrix


def test_confusion_matrix_basic():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    expected = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    assert (cm == expected).all()


def test_confusion_matrix_infer_labels_sorted():
    # 不提供 labels 时，应从 y_true/y_pred 自动推断并按升序排序
    y_true = [2, 2, 1, 0]
    y_pred = [2, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )
    assert (cm == expected).all()

