"""
Receives user form data, runs real-time feature extraction, calls the ML
pipeline, and returns AI-generated evacuation guidance.
"""

import sys
import os
import json
import logging

from flask import Blueprint, request, jsonify

ML_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "machine-learning-stuff")
)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from ai import call_ai_model
from inference import FireHazardService
from realtime_data import get_nearest_calfire_perimeter_metrics, build_point_next_hour

logger     = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)

_FIRE_SVC: FireHazardService | None = None

def _get_fire_svc() -> FireHazardService:
    global _FIRE_SVC
    if _FIRE_SVC is None:
        _FIRE_SVC = FireHazardService(
            fire_model_path=os.path.join(ML_DIR, "model_1", "fire_spread_model.pkl"),
            fire_feat_path =os.path.join(ML_DIR, "model_1", "fire_spread_features.pkl"),
            haz_model_path =os.path.join(ML_DIR, "model_2", "hazard_model.pkl"),
            haz_feat_path  =os.path.join(ML_DIR, "model_2", "hazard_features.pkl"),
        )
    return _FIRE_SVC


# request logging (append-only JSONL)
SAVED_INPUTS_FILE = os.path.join(os.path.dirname(__file__), "saved_inputs.jsonl")

def _persist_input(data: dict) -> None:
    """Silently append request data to a JSONL log file."""
    try:
        with open(SAVED_INPUTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to persist predict input")


# route finder based on checkboxes

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # validate required fields 
    if data.get("latitude") is None or data.get("longitude") is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (TypeError, ValueError):
        return jsonify({"error": "latitude and longitude must be numbers"}), 400

    # optional user-context fields 
    ctx = {
        **data,
        "has_disability":  bool(data.get("has_disability",  False)),
        "has_pets":        bool(data.get("has_pets",        False)),
        "has_kids":        bool(data.get("has_kids",        False)),
        "has_medications": bool(data.get("has_medications", False)),
        "other_concerns":  str(data.get("other_concerns",  "") or ""),
    }

    _persist_input(ctx)

    # real-time data: nearest CAL FIRE perimeter 
    try:
        pm = get_nearest_calfire_perimeter_metrics(lat=lat, lon=lon, search_km=100)
    except Exception as exc:
        logger.exception("CAL FIRE perimeter lookup failed")
        return jsonify({"error": f"No nearby fire perimeter found: {exc}"}), 400

    try:
        point = build_point_next_hour(lat=lat, lon=lon, pm=pm)
    except Exception as exc:
        logger.exception("build_point_next_hour failed")
        return jsonify({"error": f"Failed to build realtime features: {exc}"}), 500

    # model inference 
    try:
        ml_out = _get_fire_svc().predict_one(point)
    except Exception as exc:
        logger.exception("ML inference failed")
        return jsonify({"error": f"ML inference failed: {exc}"}), 500

    ai_response = call_ai_model({**ctx, "ml": ml_out})

    return jsonify({
        "selected_fire": {
            "fire_name": pm.get("fire_name"),
            "year":      pm.get("year"),
            "inc_num":   pm.get("inc_num"),
            "irwinid":   pm.get("irwinid"),
        },
        "ml":       ml_out,
        "guidance": ai_response,
    })