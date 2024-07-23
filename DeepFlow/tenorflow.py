from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomNormal
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



fnn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

fnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fnn.fit(X_train, y_train, epochs=5)

initial_acc = fnn.history.history['accuracy'][0]
print(f"Initial Training Accuracy: {initial_acc:.2f}")
