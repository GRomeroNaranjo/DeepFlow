import numpy as np
import loss_methods as lm
import layers
import matplotlib.pyplot as plt

def predict(self, network, inputs):
    output = inputs
    for layer in network:
        output = layer.forward(output)
    return np.argmax(output, axis=1)

def calculate_accuracy(self, network, X, y):
    predictions = self.predict(network, X)
    accuracy = np.mean(predictions == y)
    return accuracy
    
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
    
    def compile(self, learning_rate, loss):
        self.learning_rate = learning_rate
        self.loss = loss

    def fit(self, network, X_train, y_train, epochs):
        metrics_list = []
        loss_list = []
        self.network = network
        for epoch in range(epochs):
            output = X_train
            for layer in network:
                output = layer.forward(output)
            
            if self.loss.lower() == 'binary_crossentropy':
                loss_gradient = lm.binary_crossentropy(output, y_train)
            elif self.loss.lower() == 'crossentropy':
                loss_gradient = lm.crossentropy(output, y_train)
            elif self.loss.lower() == 'mean_squared_error':
                loss_gradient = lm.mean_squared_error(output, y_train)
            else:
                raise NameError("Loss function is invalid")
            
            for layer in reversed(network):
                loss_gradient = layer.backward(loss_gradient, self.learning_rate)
            
            accuracy = self.calculate_accuracy(X_train, y_train)
            print(f'Epoch: 1/1, {epoch + 1} / {epochs} [==========] Accuracy: {accuracy}')
            
            metrics_list.append(accuracy)

        return metrics_list
    
    

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
           
''' 
import numpy as np
from tensorflow.keras import datasets

class Image_Dreamer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.network = None

    def set(self, n_x1, n_y1, n_y2):
        self.network = [
            layers.Dense(n_x1, n_y1),
            layers.Relu(),
            layers.Dense(n_y1, n_y2),
            layers.Softmax()
        ]

    def train(self, learning_rate, epochs):
        fnn = Sequential()
        fnn.compile(learning_rate, 'crossentropy')
        fnn.fit(self.network, self.X_train, self.y_train, epochs=epochs)

    def imagine(self, image):
        output = self.network[0].forward(image)
        output = self.network[1].forward(output)
        output = self.network[2].forward(output)
        output = np.expand_dims(output, axis=0)
        return output

from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

dreamer = Image_Dreamer(X_train, y_train)
dreamer.set(784, 784, 10)
dreamer.train(0.3, 5)
dreamed = dreamer.imagine(X_train[1])

plt.imshow(dreamed.reshape(10, 1), cmap='gray')
plt.show()
'''