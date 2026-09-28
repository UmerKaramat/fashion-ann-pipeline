from tensorflow.keras import optimizers
import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 1. Load Parameters
with open("params.yaml","r") as f:
    config = yaml.safe_load(f)

lr = config["train"]["learning_rate"]
bs = config["train"]["batch_size"]
dr = config["train"]["dropout_rate"]
epochs = config["train"]["epochs"]

# 2. Load Processed Data
X_test = np.load("data/processed/X_test.npy")
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_test = np.load("data/processed/y_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

# 3. Define Network Architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu'),
    layers.Dropout(dr),
    layers.Dense(10,activation='softmax')
])

# 4. Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr) , 
    loss = 'sparse_categorical_crossentropy' ,
    metrics = ['accuracy']
)

# 5. Train Model 
history = model.fit(
    X_train,
    y_train,
    batch_size = bs,   
    epochs = epochs,
    validation_data = (X_val, y_val) 
)

# 6. Save Training History Logs
history_df = pd.DataFrame(history.history)
os.makedirs("models", exist_ok=True)
csv_path = os.path.join("models", "history.csv")
history_df.to_csv(csv_path, index=False)

# 7. Create model/ directory and save the model
model_path = os.path.join("models", "model.h5")
model.save(model_path)

