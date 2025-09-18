from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os

app = Flask(__name__)

# Paths to your saved model and scaler
MODEL_PATH = "diabetes_model.h5"
SCALER_PATH = "scaler.pkl"

# Make sure these files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Place {MODEL_PATH} and {SCALER_PATH} in the project folder (same folder as app.py).")

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))

FEATURE_ORDER = ["pregnancies", "glucose", "bloodpressure", "skinthickness", "insulin", "bmi", "dpf", "age"]

@app.route("/")
def home():
    return render_template("index.html")

def parse_form(form):
    arr = []
    for key in FEATURE_ORDER:
        v = form.get(key)
        if v is None or v == "":
            raise ValueError(f"Missing value for {key}")
        arr.append(float(v))
    return np.array([arr])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_arr = parse_form(request.form)
        input_scaled = scaler.transform(input_arr)
        prob = float(model.predict(input_scaled)[0][0])
        pred = 1 if prob > 0.5 else 0
        label = "⚠️ High Risk of Diabetes" if pred == 1 else "✅ Low Risk of Diabetes"
        prob_pct = f"{prob*100:.1f}%"
        return render_template("index.html", prediction=label, probability=prob_pct)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run()

