import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

# Load Model
MODEL_PATH = "saved_model/plastic_detector.h5"
model = load_model(MODEL_PATH)

# Classes = preprocess + training
class_names = ["non_plastic", "plastic"]

# Flask App
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction Function
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]           # [prob_non_plastic, prob_plastic]
    predicted_class = np.argmax(prediction)      # 0 or 1
    class_name = class_names[predicted_class]
    confidence = float(np.max(prediction)) * 100
    return class_name, confidence

# Routes
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

# Run Server
if __name__ == "__main__":
    app.run(debug=True)
