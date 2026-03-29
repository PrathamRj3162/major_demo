"""
Model Loader — Singleton Pattern
=================================
Loads the PneumoniaNet model once and caches it for reuse across
all API requests. Falls back to ImageNet pretrained weights if
no fine-tuned checkpoint exists.
"""

import os
import torch
from models.densenet_model import PneumoniaNet
from config import MODEL_WEIGHTS_PATH, NUM_CLASSES

# Module-level singleton
_model = None
_device = None


def get_device():
    """Returns the best available device (GPU if available, else CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model():
    """
    Load and cache the PneumoniaNet model.
    
    - If a fine-tuned checkpoint exists at MODEL_WEIGHTS_PATH, loads it.
    - Otherwise, initializes with ImageNet pretrained weights (transfer learning).
    - The model is set to eval mode and moved to the appropriate device.
    
    Returns:
        model (PneumoniaNet): The loaded model ready for inference.
    """
    global _model

    if _model is not None:
        return _model

    device = get_device()
    print(f"[Model Loader] Using device: {device}")

    # Initialize model with pretrained backbone
    _model = PneumoniaNet(num_classes=NUM_CLASSES, pretrained=True)

    # Load fine-tuned weights if available
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[Model Loader] Loading fine-tuned weights from: {MODEL_WEIGHTS_PATH}")
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        _model.load_state_dict(state_dict)
        print("[Model Loader] Fine-tuned weights loaded successfully.")
    else:
        print("[Model Loader] No fine-tuned weights found. Using ImageNet pretrained backbone.")
        print(f"[Model Loader] Expected path: {MODEL_WEIGHTS_PATH}")

    _model = _model.to(device)
    _model.eval()
    
    # Apply dynamic quantization for faster CPU inference
    # This makes linear layers ~2-3x faster on Render's weak CPU
    if device.type == "cpu":
        try:
            _model = torch.quantization.quantize_dynamic(
                _model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[Model Loader] Dynamic quantization applied (INT8 linear layers).")
        except Exception as e:
            print(f"[Model Loader] Quantization skipped: {e}")
    
    print("[Model Loader] Model loaded and set to evaluation mode.")

    return _model


def get_model():
    """Get the cached model instance, loading it if necessary."""
    return load_model()
