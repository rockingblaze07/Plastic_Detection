import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib

X_train = np.load("dataset/processed/train/images.npy")
y_train = np.load("dataset/processed/train/labels.npy")
X_test  = np.load("dataset/processed/test/images.npy")
y_test  = np.load("dataset/processed/test/labels.npy")

class_names = np.load("dataset/processed/classes.npy", allow_pickle=True)
num_classes = len(class_names)

y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test))

model.save("saved_model/plastic_detector.h5")

joblib.dump(class_names, "saved_model/class_names.pkl")

print("✅ Training complete! Model and class names saved in saved_model/")
