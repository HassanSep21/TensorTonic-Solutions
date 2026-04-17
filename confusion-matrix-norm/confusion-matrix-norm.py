import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if y_true.size == 0:
        K = num_classes if num_classes is not None else 0
        return np.zeros((K, K), dtype=int)

    K = num_classes if num_classes is not None else (max(y_true.max(), y_pred.max()) + 1)

    if np.any(y_true < 0) or np.any(y_pred < 0) or \
       np.any(y_true >= K) or np.any(y_pred >= K):
        raise ValueError("Labels out of range")

    indices = y_true * K + y_pred
    confusion_matrix = np.bincount(indices, minlength=K**2).reshape((K, K))

    if normalize == 'true':
        return confusion_matrix / np.maximum(np.sum(confusion_matrix, axis=1, keepdims=True), 1e-10)
    elif normalize == 'pred':
        return confusion_matrix / np.maximum(np.sum(confusion_matrix, axis=0, keepdims=True), 1e-10)
    elif normalize == 'all':
        return confusion_matrix / np.maximum(np.sum(confusion_matrix), 1e-10)
    else:
        return confusion_matrix
    