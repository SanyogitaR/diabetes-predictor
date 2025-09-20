# app.py — production-friendly version for deployment (EC2 / Render / Gunicorn)
import os
import logging
import traceback

# reduce TensorFlow logs (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pickle

# ---------- config & paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "diabetes_model.h5"
SCALER_FILENAME = "scaler.pkl"

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILENAME)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("diabetes_app")

# ---------- Flask app ----------
# set template_folder/static_folder explicitly if you moved bg.jpg to static/
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")  # override with env in production

# ---------- quick checks & load model ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    logger.error("Required files missing. Checked paths:\n %s\n %s", MODEL_PATH, SCALER_PATH)
    raise FileNotFoundError(f"Put {MODEL_FILENAME} and {SCALER_FILENAME} in the project root (next to app.py).")

try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    logger.info("Loaded model and scaler successfully.")
except Exception as e:
    logger.exception("Failed to load model or scaler. Exiting.")
    raise

# ---------- feature order (must match training) ----------
FEATURE_ORDER = [
    "pregnancies",
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "dpf",
    "age",
]

# ---------- helpers ----------
def parse_inputs(source):
    """
    Accepts a mapping (request.form or JSON dict) and returns a numpy array shape (1, n_features).
    Raises ValueError for missing/invalid values.
    """
    arr = []
    for key in FEATURE_ORDER:
        if key not in source:
            raise ValueError(f"Missing field: {key}")
        val = source.get(key)
        if val is None or str(val).strip() == "":
            raise ValueError(f"Empty value for {key}")
        try:
            fv = float(val)
        except Exception:
            raise ValueError(f"Invalid numeric value for {key}: {val}")
        arr.append(fv)
    return np.array([arr], dtype=np.float32)

def make_prediction(input_array):
    """
    Scale input and return probability (float 0..1) and label.
    """
    scaled = scaler.transform(input_array)
    prob = float(model.predict(scaled)[0][0])
    label = "High Risk" if prob > 0.5 else "Low Risk"
    return prob, label

# ---------- routes ----------
@app.route("/", methods=["GET"])
def home():
    # If bg.jpg moved into static folder, reference in template with: url_for('static', filename='bg.jpg')
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept JSON or form-encoded
        if request.is_json:
            payload = request.get_json()
            input_arr = parse_inputs(payload)
        else:
            # request.form works for standard form submission
            input_arr = parse_inputs(request.form)

        prob, label = make_prediction(input_arr)
        prob_pct = f"{prob*100:.1f}%"

        # Return JSON for API clients
        if request.is_json:
            return jsonify({
                "prediction": label,
                "probability": prob_pct,
                "probability_float": prob
            }), 200

        # For form submission, render template with user-friendly strings
        display_label = "⚠️ High Risk of Diabetes" if label == "High Risk" else "✅ Low Risk of Diabetes"
        return render_template("index.html", prediction=display_label, probability=prob_pct)

    except ValueError as ve:
        # Input validation error -> show friendly message
        logger.warning("Validation error: %s", ve)
        msg = f"Invalid input: {ve}"
        if request.is_json:
            return jsonify({"error": msg}), 400
        return render_template("index.html", prediction=msg)

    except Exception as e:
        # Unexpected server error -> log and return friendly message
        logger.exception("Unexpected error during prediction")
        if app.debug:
            # in debug show traceback (only local testing)
            tb = traceback.format_exc()
            return render_template("index.html", prediction=f"Error: {e}", details=tb)
        friendly = "Internal server error. Please try again later."
        if request.is_json:
            return jsonify({"error": friendly}), 500
        return render_template("index.html", prediction=friendly)

@app.route("/health", methods=["GET"])
def health():
    """Simple health check for uptime/readiness."""
    return jsonify({"status": "ok"}), 200

# ---------- run (local dev) ----------
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    logger.info("Starting app (dev): host=%s port=%s debug=%s", host, port, debug)
    app.run(host=host, port=port, debug=debug)
