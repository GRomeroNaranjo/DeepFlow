import numpy as np

class Layers:
    def forward(self, inputs):
        pass
    def backward(self, output_gradient, learning_rate):
        pass
    

class Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(self.inputs.T, output_gradient)
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.input_gradient = np.dot(output_gradient, self.weights.T)
        
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

        return self.input_gradient


class Relu(Layers):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * np.where(self.inputs > 0, 1, 0)

class Linear(Layers):
    def forward(self, inputs):
        self.inputs = inputs
        return inputs
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient

        
class Softmax(Layers):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        return output_gradient