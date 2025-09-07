import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Paths
OUTPUT_DIR = "dataset/processed"
MODEL_PATH = "saved_model/model.h5"

# Load data
X_test = np.load(os.path.join(OUTPUT_DIR, "test/images.npy"))
y_test = np.load(os.path.join(OUTPUT_DIR, "test/labels.npy"))

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Encode labels
encoder = LabelEncoder()
y_test_enc = encoder.fit_transform(y_test)

# Normalize images
X_test = X_test.astype("float32") / 255.0

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Reports
print("✅ Evaluation Results:")
print(classification_report(y_test_enc, y_pred_classes, target_names=encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred_classes))
