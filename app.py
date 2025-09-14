import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import joblib

# ======================
# Load Model + Encoder
# ======================
MODEL_PATH = "saved_model/plastic_detector.h5"
ENCODER_PATH = "saved_model/label_encoder.pkl"

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# ======================
# Flask App
# ======================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================
# Preprocessing Function
# ======================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))          # same size as training
    img = img / 255.0                          # normalize
    return np.expand_dims(img, axis=0)         # add batch dimension

# ======================
# Prediction Function
# ======================
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]         # softmax output
    predicted_class = np.argmax(prediction)    # index (0 or 1)
    class_name = encoder.classes_[predicted_class]  # map back to label
    confidence = float(np.max(prediction)) * 100
    return class_name, confidence

# ======================
# Routes
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run prediction
    label, confidence = predict_image(file_path)

    return jsonify({
        "prediction": label,
        "confidence": f"{confidence:.2f}%"
    })

# ======================
# Run Server
# ======================
if __name__ == "__main__":
    app.run(debug=True)
