"""
/api/predict  --  receives user form data, calls the ML model, returns results.
"""

import sys
import os
from flask import Blueprint, request, jsonify
import json

# ── Add the ML folder to the Python path so we can import from it ────
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "machine-learning-stuff"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

# ── Import your trained model wrapper ────────────────────────────────
# Uncomment the line below once you've added your real model in
# machine-learning-stuff/model.py
#
# from model import predict as ml_predict

predict_bp = Blueprint("predict", __name__)

# Module-level variable to hold the last received predict input (in-memory)
LAST_PREDICT_INPUT = None
# File to persist inputs (append-only, newline-delimited JSON). Change path if needed.
SAVED_INPUTS_FILE = os.path.join(os.path.dirname(__file__), "saved_inputs.jsonl")


@predict_bp.before_request
def _capture_predict_request():
    """Capture and persist incoming JSON for the /predict endpoint
    """
    global LAST_PREDICT_INPUT
    try:
        # Only capture POST to this blueprint's /predict route
        if request.method != "POST" or not request.path.endswith("/predict"):
            return
        data = request.get_json(silent=True)
        if data is None:
            return
        LAST_PREDICT_INPUT = data
        # Ensure directory exists then append newline-delimited JSON
        try:
            with open(SAVED_INPUTS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception:
            # Swallow file IO errors so we don't break the request flow
            pass
    except Exception:
        # Guard against unexpected errors in the hook
        pass


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "latitude": float | null,
        "longitude": float | null,
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
    has_disability = data.get("has_disability", False)
    has_pets = data.get("has_pets", False)
    has_kids = data.get("has_kids", False)
    has_medications = data.get("has_medications", False)
    other_concerns = data.get("other_concerns", "")

    # ── Call ML model (placeholder) ──────────────────────────────
    # Replace this block with your real model call, e.g.:
    #   result = ml_predict(latitude, longitude, has_disability, ...)
    #
    # For now we return static placeholder text so you can verify
    # the full pipeline works end-to-end before plugging in the model.

    fire_risk = "MODERATE"
    evacuation_route = (
        f"Head north on US-101 from ({latitude}, {longitude}). "
        "Take exit 42B toward Safe Zone Shelter A."
    )

    notes_parts = []
    if has_disability:
        notes_parts.append("Accessible vehicle requested.")
    if has_pets:
        notes_parts.append("Pet-friendly shelter assigned.")
    if has_kids:
        notes_parts.append("Family shelter with childcare available.")
    if has_medications:
        notes_parts.append("Route passes by open pharmacy.")
    if other_concerns:
        notes_parts.append(f"User note: {other_concerns}")

    notes = " ".join(notes_parts) if notes_parts else "No special accommodations needed."

    return jsonify({
        "fire_risk": fire_risk,
        "evacuation_route": evacuation_route,
        "notes": notes,
    })
