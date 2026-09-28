import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

with open("params.yaml","r") as f:
    config = yaml.safe_load(f)

X_train = np.load("data/raw/X_train.npy")
y_train = np.load("data/raw/y_train.npy")
X_test = np.load("data/raw/X_test.npy")
y_test = np.load("data/raw/y_test.npy")

test_size = config["preprocess"]["test_size"]
seed = config["preprocess"]["seed"]

X_train = X_train.astype('float32') / 255.0 * 0.9
X_test = X_test.astype('float32') / 255.0 * 0.9

X_train , X_val , y_train , y_val = train_test_split(X_train , y_train , test_size = test_size , random_state = seed)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)



