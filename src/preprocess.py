import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion.npz")

x = (d["x_train"].astype("float32") / 255.0 - 0.2860) / 0.3530  # teammate-sim: z-score standardization
x_test = (d["x_test"].astype("float32") / 255.0 - 0.2860) / 0.3530  # teammate-sim: z-score standardization

x_train, x_val, y_train, y_val = train_test_split(
    x, d["y_train"], test_size=p["test_size"], random_state=p["seed"], stratify=d["y_train"]
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=d["y_test"],
)
print("train/val/test:", x_train.shape, x_val.shape, x_test.shape)# TODO: try different split ratios
# TODO: add shuffle check
