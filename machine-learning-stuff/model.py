"""ML model wrapper used by the Flask prediction route."""

from datetime import datetime
import os

import joblib
import numpy as np


ML_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ML_DIR, "ca_wildfire_model.joblib")
SCALER_PATH = os.path.join(ML_DIR, "ca_wildfire_scaler.joblib")


MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH) if os.path.isfile(SCALER_PATH) else None


def _to_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _to_int_bool(value):
    return 1 if bool(value) else 0


def _parse_datetime(date_str, time_str):
    if date_str and time_str:
        try:
            return datetime.fromisoformat(f"{date_str}T{time_str}")
        except ValueError:
            pass
    return datetime.now()


def _risk_label(raw_prediction, probability):
    text_pred = str(raw_prediction).strip().upper()
    if text_pred in {"LOW", "MODERATE", "HIGH"}:
        return text_pred

    numeric_pred = _to_float(raw_prediction, fallback=0.0)
    score = probability if probability is not None else numeric_pred
    if score >= 0.67:
        return "HIGH"
    if score >= 0.34:
        return "MODERATE"
    return "LOW"


def _build_feature_vector(
    latitude,
    longitude,
    event_dt,
    has_disability,
    has_pets,
    has_kids,
    has_medications,
):
    base_features = [
        _to_float(latitude),
        _to_float(longitude),
        float(event_dt.month),
        float(event_dt.day),
        float(event_dt.hour),
        float(event_dt.minute),
        float(event_dt.weekday()),
        float(1 if event_dt.weekday() >= 5 else 0),
        float(_to_int_bool(has_disability)),
        float(_to_int_bool(has_pets)),
        float(_to_int_bool(has_kids)),
        float(_to_int_bool(has_medications)),
    ]

    n_features = int(getattr(MODEL, "n_features_in_", len(base_features)))
    if n_features <= len(base_features):
        return np.array(base_features[:n_features], dtype=float)

    padded = base_features + [0.0] * (n_features - len(base_features))
    return np.array(padded, dtype=float)


def predict(
    latitude,
    longitude,
    date,
    time,
    location_name,
    has_disability,
    has_pets,
    has_kids,
    has_medications,
    other_concerns,
):
    event_dt = _parse_datetime(date, time)
    feature_vector = _build_feature_vector(
        latitude,
        longitude,
        event_dt,
        has_disability,
        has_pets,
        has_kids,
        has_medications,
    )

    model_input = feature_vector.reshape(1, -1)
    if SCALER is not None:
        model_input = SCALER.transform(model_input)

    raw_prediction = MODEL.predict(model_input)[0]
    probability = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(model_input)[0]
        probability = float(np.max(probabilities))

    fire_risk = _risk_label(raw_prediction, probability)

    route_direction = "north" if _to_float(latitude) <= 36.0 else "south"
    evacuation_route = (
        f"Move {route_direction} from ({_to_float(latitude):.4f}, {_to_float(longitude):.4f}) "
        "to the nearest designated shelter on your county evacuation map."
    )

    notes_parts = [
        f"Model prediction: {raw_prediction}",
        f"Event time: {event_dt.isoformat(timespec='minutes')}",
    ]
    if location_name:
        notes_parts.append(f"Location: {location_name}")
    if has_disability:
        notes_parts.append("Accessibility support needed")
    if has_pets:
        notes_parts.append("Pet-friendly shelter advised")
    if has_kids:
        notes_parts.append("Family/child support considered")
    if has_medications:
        notes_parts.append("Medication continuity required")
    if other_concerns:
        notes_parts.append(f"Other concerns: {other_concerns}")

    model_output = {
        "raw_prediction": str(raw_prediction),
        "probability": probability,
        "feature_count": int(model_input.shape[1]),
    }
    if hasattr(MODEL, "classes_"):
        model_output["classes"] = [str(label) for label in MODEL.classes_]

    return {
        "fire_risk": fire_risk,
        "evacuation_route": evacuation_route,
        "notes": ". ".join(notes_parts),
        "model_output": model_output,
        "request_context": {
            "latitude": _to_float(latitude),
            "longitude": _to_float(longitude),
            "location_name": location_name or "",
            "date": event_dt.date().isoformat(),
            "time": event_dt.time().strftime("%H:%M"),
        },
    }
