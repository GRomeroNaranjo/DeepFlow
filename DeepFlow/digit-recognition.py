import numpy as np
from tensorflow.keras import datasets
from DeepFlow import layers, models
import matplotlib.pyplot as plt
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255

network = [
    layers.Dense(784, 128),
    layers.Relu(),
    layers.Dense(128, 10),
    layers.Softmax()
]

nn = models.Sequential()
nn.compile(learning_rate=0.5, loss='mean_squared_error', optimizer='mbgd')
accuracy_train, loss_list = nn.fit(network, X_train, y_train, epochs=25)

plt.plot(accuracy_train)
plt.show()