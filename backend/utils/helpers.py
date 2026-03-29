"""
Helper Utilities
================
Common utility functions used across the backend:
  - Base64 image encoding/decoding
  - File path helpers
  - Response formatting
"""

import os
import io
import base64
import uuid
from PIL import Image
from datetime import datetime


def image_to_base64(pil_image: Image.Image, format: str = "JPEG", quality: int = 75) -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        pil_image: PIL Image object.
        format: Image format (JPEG for speed, PNG for lossless).
        quality: JPEG quality (1-100). 85 is good balance of quality/size.

    Returns:
        Base64 string with data URI prefix for direct embedding in HTML/React.
    """
    buffer = io.BytesIO()
    # Ensure RGB mode for JPEG (JPEG doesn't support RGBA)
    if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P", "LA"):
        pil_image = pil_image.convert("RGB")
    save_kwargs = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    pil_image.save(buffer, **save_kwargs)
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode("utf-8")
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64_string}"


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename to avoid collisions in the uploads folder.

    Args:
        original_filename: The original uploaded filename.

    Returns:
        A unique filename string.
    """
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"xray_{timestamp}_{unique_id}.{ext}"


def format_prediction_response(prediction, confidence, needs_review, gradcam_b64=None):
    """
    Format a standardized prediction API response.

    Args:
        prediction: Class name string (e.g., "NORMAL" or "PNEUMONIA").
        confidence: Confidence score as float (0-1).
        needs_review: Boolean indicating if confidence is below threshold.
        gradcam_b64: Optional base64 Grad-CAM image.

    Returns:
        dict: Formatted response dictionary.
    """
    response = {
        "prediction": prediction,
        "confidence": round(float(confidence) * 100, 2),
        "confidence_raw": round(float(confidence), 4),
        "needs_review": needs_review,
        "status": "needs_review" if needs_review else "confirmed",
        "timestamp": datetime.now().isoformat()
    }
    if gradcam_b64:
        response["gradcam_image"] = gradcam_b64
    return response
