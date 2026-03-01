"""
/api/predict  --  receives user form data, calls the ML model, returns results.
"""

import sys
import os
from flask import Blueprint, request, jsonify

# ── Add the ML folder to the Python path so we can import from it ────
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "machine-learning-stuff"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from model import predict as ml_predict

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "latitude": float | null,
        "longitude": float | null,
        "location_name": str,
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "has_disability": bool,
        "has_pets": bool,
        "has_kids": bool,
        "has_medications": bool,
        "other_concerns": str
    }

    Returns JSON with fire_risk, evacuation_route, and notes.
    """

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # ── Extract fields ───────────────────────────────────────────
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    location_name = data.get("location_name", "")
    date = data.get("date")
    time = data.get("time")
    has_disability = data.get("has_disability", False)
    has_pets = data.get("has_pets", False)
    has_kids = data.get("has_kids", False)
    has_medications = data.get("has_medications", False)
    other_concerns = data.get("other_concerns", "")

    try:
        result = ml_predict(
            latitude=latitude,
            longitude=longitude,
            date=date,
            time=time,
            location_name=location_name,
            has_disability=has_disability,
            has_pets=has_pets,
            has_kids=has_kids,
            has_medications=has_medications,
            other_concerns=other_concerns,
        )
    except Exception as exc:
        return jsonify({"error": f"Model prediction failed: {str(exc)}"}), 500

    return jsonify(result)
