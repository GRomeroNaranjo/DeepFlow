import numpy as np
  
def binary_crossentropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = y_pred - y_true
    return loss, gradient

def crossentropy(y_pred, y_true):
    samples = len(y_pred)
    y_true = np.eye(samples)[y_true]
    return (y_pred - y_true) / samples

def mean_squared_error(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    squared_diff = (y_pred - y_true) ** 2
    loss = np.mean(squared_diff)
    return loss
