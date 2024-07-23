import numpy as np
import matplotlib.pyplot as plt

def predict(network, inputs):
    output = inputs
    for layer in network:
        output = layer.forward(output)
    return np.argmax(output, axis=1)

def calculate_accuracy(network, X, y):
    predictions = predict(network, X)
    accuracy = np.mean(predictions == y)
    return accuracy

def create_mini_batches(X, y, batch_size):
    data = np.hstack((X, y.reshape(-1, 1)))
    mini_batches = [data[k:k + batch_size] for k in range(0, len(data), batch_size)]
    mini_batches = [(batch[:, :-1], batch[:, -1]) for batch in mini_batches]
    return mini_batches

def calculate_loss_gradient(y_pred, y_true):
    samples = len(y_pred)
    y_pred_int = y_pred.astype(np.int32)
    y_true_int = y_true.astype(np.int32)
    
    y_true = np.eye(y_pred_int.shape[1])[y_true_int]

    if len(y_pred.shape) == 1:
        y_pred = np.expand_dims(y_pred, axis=1)

    loss = y_pred - y_true
    loss = loss / samples
    
    return loss

def binary_crossentropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss_gradient = calculate_loss_gradient(y_pred, y_true)
    return loss, loss_gradient

def mean_squared_error(y_pred, y_true):
    samples = len(y_pred)
    loss_gradient = calculate_loss_gradient(y_pred, y_true)
    loss = loss_gradient ** 2
    loss = loss / samples
    
    return loss, loss_gradient

class Sequential:
    def predict(self, inputs):
        output = inputs
        for layer in self.network:
            output = layer.forward(output)
        return np.argmax(output, axis=1)

    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def compile(self, learning_rate, loss, optimizer):
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        
    def fit(self, network, X_train, y_train, epochs):
        self.network = network
        if self.optimizer.lower() == 'gd':
            accuracy_list = []
            loss_list = []
            for epoch in range(epochs):
                output = X_train
                for layer in network:
                    output = layer.forward(output)
                
                if self.loss.lower() == 'binary_crossentropy':
                    loss, loss_gradient = binary_crossentropy(output, y_train)
                elif self.loss.lower() == 'mean_squared_error':
                    loss, loss_gradient = mean_squared_error(output, y_train)
                else:
                    raise ValueError("Loss function is invalid")
                
                loss_list.append(loss)
                accuracy_list.append(accuracy)
                
                for layer in reversed(network):
                    loss_gradient = layer.backward(loss_gradient, self.learning_rate)
                
                accuracy = self.calculate_accuracy(X_train, y_train)
                print(f'Epoch: {epoch + 1}/{epochs} [=================] Accuracy: {accuracy:.4f}')
                
            return accuracy_list, loss_list
        
        elif self.optimizer.lower() == 'mbgd':
            accuracy_list = []
            loss_list = []

            for epoch in range(epochs):
                accuracy = []
                mini_batches = create_mini_batches(X_train, y_train, 64)

                for batch_X, batch_y in mini_batches:
                    output = batch_X
                    for layer in network:
                        output = layer.forward(output)

                    if self.loss.lower() == 'binary_crossentropy':
                        loss, loss_gradient = binary_crossentropy(output, batch_y)
                    elif self.loss.lower() == 'mean_squared_error':
                        loss, loss_gradient = mean_squared_error(output, batch_y)
                    else:
                        raise ValueError("Loss function is invalid")

                    accuracy.append(self.calculate_accuracy(batch_X, batch_y))

                    for layer in reversed(network):
                        loss_gradient = layer.backward(loss_gradient, self.learning_rate)

                print(f'Epoch: {epoch + 1}/{epochs} [=================] Accuracy: {np.mean(accuracy)}')

                accuracy_list.append(np.mean(accuracy))
                loss_list.append(loss)

            return accuracy_list, loss_list
        else:
            raise ValueError("Optimizer is invalid")
        

class Logistic_Regression:
    def __init__(self, X_train, y_train, n_inputs, learning_rate, epochs):
        self.epochs = epochs
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        
        self.weights = np.random.randn(n_inputs, 1)
        self.biases = np.random.randn(1, 1)
      
    def dense_forward(self):
        self.output = np.dot(self.X_train, self.weights) + self.biases
        return self.output
    
    def sigmoid_forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def calculate_loss(self, y_pred, y_true):
        loss, gradient = lm.binary_crossentropy(y_pred, y_true)
        return loss, gradient
    
    def dense_backward(self, inputs, gradients):
        self.weights_gradient = np.dot(inputs.T, gradients)
        self.bias_gradient = np.sum(gradients, axis=0, keepdims=True)
        
        self.weights -= self.learning_rate * self.weights_gradient
        self.biases -= self.learning_rate * self.bias_gradient
        
    def predict(self, X):
        output = np.dot(X, self.weights) + self.biases
        output = self.sigmoid_forward(output)
        return output
    
    def calculate_accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        predicted_classes = (predictions >= 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y_test)
        return accuracy
        
    def train(self):
        for epoch in range(self.epochs):
            output = self.dense_forward()
            output = self.sigmoid_forward(output)
            
            loss, gradients = self.calculate_loss(output, self.y_train)
            self.dense_backward(self.X_train, gradients)
            
            metrics = self.calculate_accuracy(self.X_train, self.y_train)
            print(f'Epoch: {epoch + 1}/{self.epochs} Loss: {loss:.4f} Accuracy: {metrics:.4f}')
          