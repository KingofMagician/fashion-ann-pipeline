import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = model.predict(d["x_test"]).argmax(axis=1)

os.makedirs("reports", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot(cmap="Blues")
plt.savefig("reports/confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
print("test accuracy:", acc)