import numpy as np
from src.evaluate import compute_direction, plot_confusion_matrix

def test_confusion_matrix_shape():
    true = np.array([1,2,3,4,5])
    pred = np.array([1,2,2,4,6])

    fig = plot_confusion_matrix(true, pred)

    assert fig is not None
