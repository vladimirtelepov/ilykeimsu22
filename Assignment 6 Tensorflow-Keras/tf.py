import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


class Sin(layers.Layer):
    def call(self, x):
        return keras.backend.sin(x)


class Max(layers.Layer):
    def call(self, x):
        return keras.backend.max(x)


initializer = tf.keras.initializers.Identity()
model = keras.Sequential(
    [
        layers.Dense(10, kernel_initializer=initializer),
        layers.Dense(20, kernel_initializer=initializer),
        layers.Dense(20, kernel_initializer=initializer),
        layers.Dropout(0.2),
        layers.ReLU(),
        layers.Reshape((4, 5)),
        layers.Dense(10, kernel_initializer=initializer),
        layers.ReLU(),
        layers.Flatten(),
        Sin(),
        layers.Dropout(0.1),
        layers.Dense(3, kernel_initializer=initializer),
        layers.Softmax(),
        Max()
    ]
)

data = np.arange(10).reshape((1,-1))
x_tf = tf.convert_to_tensor(data, np.float32)
y = model(x_tf)
print(y)