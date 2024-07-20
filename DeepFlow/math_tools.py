import numpy as np

def binary_crossentropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = y_pred - y_true
    return loss, gradient

def crossentropy(y_pred, y_true):
    samples = len(y_pred)
    y_true = np.eye(y_pred.shape[1])[y_true]
    return (y_pred - y_true) / samples

def gradient_descent(inputs, weights, biases, output_gradient, learning_rate):
    weights_gradient = np.dot(output_gradient, inputs)
    bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
    inputs_gradient = np.dot(weights, output_gradient)
    
    weights -= learning_rate * weights_gradient
    biases -= learning_rate * bias_gradient
    
    return inputs_gradient, weights, biases
    

def one_hot_encoder(inputs, samples):
    one_hot_encoded = np.eye(samples)[inputs]
    return one_hot_encoded

class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        output = np.max(0, self.inputs)
        return output
    
    def backward(self, output_gradient, learning_rate):
        self.input_gradient = output_gradient * np.where(self.inputs > 0, 1, 0)
        return self.input_gradient
    
class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exp_sum = np.sum(exp_values)
        probability_distribution = exp_values / exp_sum
        self.probability_distribution = probability_distribution
        return probability_distribution
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        return input_gradient
    
def initialiser(input_size):
    initialiser = np.sqrt(2. / input_size)
    return initialiser

    



