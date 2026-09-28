import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Load the trained model and test datasets
print("Loading model and test data...")
model_path = os.path.join("models", "model.h5")
model = keras.models.load_model(model_path)

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# 2. Compute test loss and accuracy
print("Evaluating model performance on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# 3. Generate predictions for the confusion matrix
print("Generating predictions...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute raw confusion matrix values
cm = confusion_matrix(y_test, y_pred)

# 4. Generate and save the Confusion Matrix plot
print("Creating confusion matrix image...")
os.makedirs("metrics", exist_ok=True)  
cm_image_path = os.path.join("metrics", "confusion_matrix.png")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Fashion Pipeline')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(cm_image_path)
plt.close()
print(f"Confusion matrix saved to: {cm_image_path}")

# 5. Write all metrics to metrics.json at the project root
metrics_data = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc)
}

metrics_json_path = "metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_data, f, indent=4)

print(f"Metrics JSON saved to root: {metrics_json_path}")
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
