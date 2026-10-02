import os
import numpy as np
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
os.makedirs("data/raw", exist_ok=True)
np.savez_compressed("data/raw/fashion.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print("Saved raw data:", x_train.shape, x_test.shape)