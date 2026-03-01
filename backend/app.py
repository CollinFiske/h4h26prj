"""
CA Fire Detection & Evacuation -- Flask Backend
================================================
Entry point for the Flask development server.

Run:
    cd backend
    python app.py
"""

from flask import Flask
from flask_cors import CORS
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
