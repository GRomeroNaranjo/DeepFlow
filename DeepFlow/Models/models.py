import numpy as np
from DeepFlow.Models.loss_methods import sparse_categorical_crossentropy_prime

def predict(network, inputs):
    output = inputs
    for layer in network:
        output = layer.forward(output)
    return np.argmax(output, axis=1)

def calculate_accuracy(network, X, y):
    predictions = predict(network, X)
    accuracy = np.mean(predictions == y)
    return accuracy

class Sequential:
    def compile(self, learning_rate, loss, metrics):
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
    
    def fit(self, network, X_train, y_train, epochs):
        for epoch in range(epochs):
            output = X_train
            for layer in network:
                output = layer.forward(output)
            
            if self.metrics.lower() == 'accuracy':
                metrics = calculate_accuracy(network, X_train, y_train)
            
            if self.loss.lower() == 'sparse_categorical_crossentropy':
                loss_gradient = sparse_categorical_crossentropy_prime(network[-1].output, y_train)
                
            for layer in reversed(network):
                loss_gradient = layer.backward(loss_gradient, self.learning_rate)
            
            print(f'Epoch: {epoch + 1} ======> Metrics: {metrics * 100}%')