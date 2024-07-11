import numpy as np
from tensorflow.keras import datasets
from DeepFlow.Layers.layers import Dense, Relu, Softmax
from DeepFlow.Models.models import Sequential as Sequential

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

network = [
    Dense(784, 128),
    Relu(), 
    Dense(128, 10), 
    Softmax()
]

nn = Sequential()
nn.compile(learning_rate=0.1, loss='sparse_categorical_crossentropy', metrics='accuracy')
nn.fit(network, X_train, y_train, 400)
