from DeepFlow import layers, models
from tensorflow.keras import datasets

class Neural_Network:
    def __init__(self, learning_rate, loss, optimizer, epochs):
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        
    def train(self):
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

        self.network = [
            layers.Dense(3072, 1024),
            layers.Relu(),
            layers.Dense(1024, 10),
            layers.Softmax()
        ]
        
        fnn = models.Sequential()
        fnn.compile(learning_rate=self.learning_rate, loss=self.loss, optimizer=self.optimizer)
        accuracy, loss = fnn.fit(self.network, X_train, y_train, epochs=self.epochs)
        return accuracy, loss
    
    def predict(self, data):
        data = data.reshape(data.shape[0], -1).astype('float32') / 255
        prediction = self.network(data)
        return prediction
    
    def calculate_accuracy(self, y_pred, y_true):
        accuracy = (y_pred.argmax(axis=1) == y_true).mean()
        return accuracy
    
network = Neural_Network(0.5, 'mean_squared_error', 'mbgd', 5)
accuracy, loss = network.train()
print(f'Accuracy: {accuracy}, Loss: {loss}')
