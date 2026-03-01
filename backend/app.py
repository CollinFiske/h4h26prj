"""
CA Fire Detection & Evacuation -- Flask Backend
================================================
Entry point for the Flask development server.

Run:
    cd backend
    python app.py
"""

import os
import sys

from flask import Flask
from flask_cors import CORS

# ── Debug: show where Python is looking for files ────────────────
print(f"[app.py] CWD: {os.getcwd()}")
print(f"[app.py] __file__: {os.path.abspath(__file__)}")

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "machine-learning-stuff"))
print(f"[app.py] ML_DIR resolved to: {ML_DIR}")
print(f"[app.py] ML_DIR exists: {os.path.isdir(ML_DIR)}")

model_file = os.path.join(ML_DIR, "ca_wildfire_model.joblib")
scaler_file = os.path.join(ML_DIR, "ca_wildfire_scaler.joblib")
print(f"[app.py] model joblib exists: {os.path.isfile(model_file)}")
print(f"[app.py] scaler joblib exists: {os.path.isfile(scaler_file)}")

from routes.predict import predict_bp

def create_app():
    app = Flask(__name__)
    CORS(app)  # allow React dev server on :3000

    # ── Register Blueprints ──────────────────────────────────────
    app.register_blueprint(predict_bp, url_prefix="/api")

    # ── Health Check ─────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    # debug=True gives auto-reload + detailed error pages
    app.run(host="0.0.0.0", port=5000, debug=True)
