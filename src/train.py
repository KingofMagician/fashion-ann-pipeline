import os
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

p = yaml.safe_load(open("params.yaml"))["train"]
d = np.load("data/processed/data.npz")

model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(p["dense_units"], activation="relu"),
    keras.layers.Dropout(p["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=keras.optimizers.Adam(p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

hist = model.fit(
    d["x_train"], d["y_train"],
    validation_data=(d["x_val"], d["y_val"]),
    epochs=p["epochs"], batch_size=p["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(hist.history).to_csv("models/history.csv", index=False)