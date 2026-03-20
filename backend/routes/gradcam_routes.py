"""
Grad-CAM Routes
================
Handles:
  POST /api/gradcam — Generate Grad-CAM heatmap for an uploaded image
"""

import os
from flask import Blueprint, request, jsonify
from utils.preprocessing import preprocess_image, validate_image
from utils.helpers import image_to_base64, generate_unique_filename
from services.gradcam import GradCAM
from services.prediction import predict
from config import UPLOAD_FOLDER

gradcam_bp = Blueprint("gradcam", __name__)


@gradcam_bp.route("/api/gradcam", methods=["POST"])
def generate_gradcam():
    """
    Generate Grad-CAM visualization for a chest X-ray image.
    
    Accepts:
      - multipart/form-data with 'file' field
      - JSON body with 'filename' (previously uploaded)
    
    Returns:
      - Original image (base64)
      - Grad-CAM overlay image (base64)
      - Raw heatmap image (base64)
      - Prediction and confidence
    """
    image_path = None

    # Handle file upload
    if "file" in request.files:
        file = request.files["file"]
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        unique_name = generate_unique_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(image_path)

    # Handle filename reference
    elif request.is_json and "filename" in request.json:
        filename = request.json["filename"]
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({"error": f"File '{filename}' not found"}), 404
    else:
        return jsonify({"error": "Provide a file or filename"}), 400

    try:
        # Preprocess
        image_tensor, pil_image = preprocess_image(image_path)

        # Get prediction first
        result = predict(image_tensor)

        # Generate Grad-CAM
        gradcam = GradCAM()
        overlay_img, heatmap_img = gradcam.generate_overlay(
            image_tensor, pil_image, target_class=result["predicted_index"]
        )

        return jsonify({
            "original_image": image_to_base64(pil_image),
            "gradcam_overlay": image_to_base64(overlay_img),
            "heatmap_image": image_to_base64(heatmap_img),
            "prediction": result["predicted_class"],
            "confidence": round(result["confidence"] * 100, 2),
            "target_class": result["predicted_class"]
        }), 200

    except Exception as e:
        return jsonify({"error": f"Grad-CAM generation failed: {str(e)}"}), 500
