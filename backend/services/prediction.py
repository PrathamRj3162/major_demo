"""
Prediction Service
===================
Handles the full inference pipeline:
  1. Load/access the cached model
  2. Run forward pass on preprocessed image
  3. Apply softmax to get probabilities
  4. Apply confidence threshold for decision logic
  5. Return prediction, confidence, and review status
"""

import torch
import torch.nn.functional as F
from models.model_loader import get_model, get_device
from config import CLASS_NAMES, CONFIDENCE_THRESHOLD


def predict(image_tensor):
    """
    Run inference on a preprocessed image tensor.

    Args:
        image_tensor: Preprocessed tensor of shape (1, 3, 224, 224).

    Returns:
        dict: {
            "predicted_class": str,     # "NORMAL" or "PNEUMONIA"
            "predicted_index": int,     # 0 or 1
            "confidence": float,        # 0.0 to 1.0
            "probabilities": dict,      # Per-class probabilities
            "needs_review": bool,       # True if confidence < threshold
            "threshold": float          # The confidence threshold used
        }
    """
    model = get_model()
    device = get_device()

    # Move tensor to device
    image_tensor = image_tensor.to(device)

    # Forward pass (no gradient computation needed for inference)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    # Extract predictions
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    predicted_class = CLASS_NAMES[predicted_idx]

    # Build per-class probability dict
    prob_dict = {
        CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    # Decision logic: flag low-confidence predictions for review
    needs_review = confidence < CONFIDENCE_THRESHOLD

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "needs_review": needs_review,
        "threshold": CONFIDENCE_THRESHOLD
    }
