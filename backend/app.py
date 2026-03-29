"""
AI-Driven Precision Diagnostics System — Flask Application
============================================================
Main entry point for the backend server.

This Flask application provides REST APIs for:
  - /api/upload       — Upload chest X-ray images
  - /api/predict      — Run pneumonia detection inference
  - /api/gradcam      — Generate Grad-CAM visual explanations
  - /api/federated-train — Run federated learning simulation
  - /api/model-stats  — Get model performance metrics

The application uses:
  - DenseNet121 (transfer learning) for classification
  - Grad-CAM for explainable AI
  - FedAvg for federated learning simulation
"""

from flask import Flask, jsonify
from flask_cors import CORS
from config import UPLOAD_FOLDER

# Import route blueprints
from routes.predict_routes import predict_bp
from routes.gradcam_routes import gradcam_bp
from routes.federated_routes import federated_bp
from routes.stats_routes import stats_bp

# Singleton GradCAM instance (cached across requests)
_gradcam_instance = None


def get_gradcam():
    """Get or create the singleton GradCAM instance."""
    global _gradcam_instance
    if _gradcam_instance is None:
        from services.gradcam import GradCAM
        _gradcam_instance = GradCAM()
        print("[App] GradCAM instance created and cached.")
    return _gradcam_instance


def create_app():
    """
    Application factory function.
    Creates and configures the Flask application.
    """
    app = Flask(__name__)
    
    # ─── Configuration ──────────────────────────────────
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    
    # ─── CORS ───────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # ─── Register Blueprints ────────────────────────────
    app.register_blueprint(predict_bp)
    app.register_blueprint(gradcam_bp)
    app.register_blueprint(federated_bp)
    app.register_blueprint(stats_bp)
    
    # ─── Preload Model & GradCAM on First Request ──────
    # Uses before_request to lazy-load on the first API call.
    # This avoids Gunicorn boot timeout on Render's free tier.
    _preloaded = {"done": False}
    
    @app.before_request
    def preload_model():
        if not _preloaded["done"]:
            try:
                from models.model_loader import get_model
                print("[App] Preloading DenseNet121 model...")
                get_model()
                print("[App] Model preloaded successfully!")
                
                print("[App] Initializing GradCAM...")
                get_gradcam()
                print("[App] GradCAM ready!")
            except Exception as e:
                print(f"[App] Warning: Model preload failed: {e}")
            _preloaded["done"] = True
    
    # ─── Health Check ───────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "AI Precision Diagnostics System",
            "version": "1.0.0"
        }), 200
    
    # ─── Error Handlers ─────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size: 16MB"}), 413
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500
    
    return app


# ─── Application Entry Point ───────────────────────────
if __name__ == "__main__":
    app = create_app()
    
    print("\n" + "=" * 60)
    print("  AI Precision Diagnostics System — Backend Server")
    print("  " + "-" * 54)
    print("  Endpoints:")
    print("    POST /api/upload           — Upload X-ray image")
    print("    POST /api/predict          — Run inference")
    print("    POST /api/gradcam          — Generate Grad-CAM")
    print("    POST /api/federated-train  — Federated learning")
    print("    GET  /api/model-stats      — Performance metrics")
    print("    GET  /api/health           — Health check")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5001, debug=True)
