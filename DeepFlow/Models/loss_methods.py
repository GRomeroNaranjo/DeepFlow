import numpy as np

def sparse_categorical_crossentropy_prime(y_pred, y_true):
    samples = len(y_pred)
    labels = len(y_pred[0])
    
    y_true = np.eye(labels)[y_true]
    output_gradient = y_pred - y_true
    output_gradient /= samples

    return output_gradient