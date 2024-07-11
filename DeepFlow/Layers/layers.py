import numpy as np

class Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(self.inputs.T, output_gradient)
        self.input_gradient = np.dot(output_gradient, self.weights.T)
        
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        return self.input_gradient

        
class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, output_gradient, learning_rate):
        self.input_gradient = output_gradient.copy()
        self.input_gradient[self.inputs <= 0] = 0
        return self.input_gradient
    
class Sigmoid:
    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        self.output_gradient = output_gradient
        self.input_gradient = output_gradient * (1 - output_gradient)
        return self.input_gradient
    
class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient
