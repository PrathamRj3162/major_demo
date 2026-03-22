"""
Prediction & Upload Routes
===========================
Handles:
  POST /api/upload   — Upload an X-ray image
  POST /api/predict  — Run inference on an uploaded image
"""

import os
from flask import Blueprint, request, jsonify
from utils.preprocessing import preprocess_image, validate_image
from utils.helpers import generate_unique_filename, format_prediction_response, image_to_base64
from services.prediction import predict
from services.gradcam import GradCAM
from config import UPLOAD_FOLDER

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/api/upload", methods=["POST"])
def upload_image():
    """
    Upload a chest X-ray image.
    
    Accepts multipart/form-data with a 'file' field.
    Validates the image, saves it, and returns the saved filename.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate image
    is_valid, error_msg = validate_image(file)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # Save with unique filename
    unique_name = generate_unique_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    return jsonify({
        "message": "Image uploaded successfully",
        "filename": unique_name,
        "path": save_path
    }), 200


@predict_bp.route("/api/predict", methods=["POST"])
def predict_image():
    """
    Run full inference pipeline on an uploaded image.
    
    Accepts either:
      - multipart/form-data with 'file' field (direct upload + predict)
      - JSON body with 'filename' field (predict a previously uploaded image)
    
    Returns prediction, confidence, probabilities, review status,
    and optional Grad-CAM visualization.
    """
    image_path = None

    # Option 1: Direct file upload
    if "file" in request.files:
        file = request.files["file"]
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        unique_name = generate_unique_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(image_path)

    # Option 2: Previously uploaded filename
    elif request.is_json and "filename" in request.json:
        filename = request.json["filename"]
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({"error": f"File '{filename}' not found"}), 404

    else:
        return jsonify({"error": "Provide a file upload or filename in JSON body"}), 400

    try:
        # Preprocess the image
        image_tensor, pil_image = preprocess_image(image_path)

        # Run inference
        result = predict(image_tensor)

        # Generate Grad-CAM
        gradcam = GradCAM()
        overlay_img, heatmap_img = gradcam.generate_overlay(
            image_tensor, pil_image, target_class=result["predicted_index"]
        )

        # Build response
        response = format_prediction_response(
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            needs_review=result["needs_review"],
            gradcam_b64=image_to_base64(overlay_img)
        )
        response["probabilities"] = result["probabilities"]
        response["original_image"] = image_to_base64(pil_image)
        response["heatmap_image"] = image_to_base64(heatmap_img)
        response["threshold"] = result["threshold"]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up the uploaded file to prevent disk leaks on the server
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as cleanup_err:
                print(f"Failed to clean up file {image_path}: {cleanup_err}")
