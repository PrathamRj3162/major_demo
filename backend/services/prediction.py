"""
Prediction service — takes a preprocessed image tensor, runs it through
the model, applies softmax, and returns the class + confidence.
If confidence is below the threshold, it gets flagged for manual review.
"""

import torch
import torch.nn.functional as F
from models.model_loader import get_model, get_device
from config import CLASS_NAMES, CONFIDENCE_THRESHOLD


def predict(image_tensor):
    """
    Run inference on a preprocessed image.
    Returns a dict with the predicted class, confidence, per-class
    probabilities, and whether it needs human review.
    """
    model = get_model()
    device = get_device()

    image_tensor = image_tensor.to(device)

    # no need to track gradients during inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(probabilities, dim=1)
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    predicted_class = CLASS_NAMES[predicted_idx]

    # per-class breakdown (as percentages)
    prob_dict = {
        CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    # flag low-confidence predictions
    needs_review = confidence < CONFIDENCE_THRESHOLD

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "needs_review": needs_review,
        "threshold": CONFIDENCE_THRESHOLD
    }
