"""
Image Preprocessing Pipeline
=============================
Handles all image transformations needed before model inference:
  1. Load image from file path or PIL Image
  2. Resize to 224×224
  3. Convert to RGB
  4. Normalize with ImageNet mean/std
  5. Convert to PyTorch tensor with batch dimension

These transforms match the preprocessing used during DenseNet121
training on ImageNet, ensuring consistent feature extraction.
"""

import io
import numpy as np
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transform():
    """
    Returns the standard preprocessing transform pipeline.
    This matches ImageNet preprocessing for DenseNet121.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image_source):
    """
    Preprocess an image for model inference.

    Args:
        image_source: Can be a file path (str), PIL Image, or bytes.

    Returns:
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        pil_image: The original PIL image (RGB) for Grad-CAM overlay.
    """
    # Load image based on source type
    if isinstance(image_source, str):
        pil_image = Image.open(image_source)
    elif isinstance(image_source, bytes):
        pil_image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, Image.Image):
        pil_image = image_source
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    # Convert to RGB (medical images may be grayscale)
    pil_image = pil_image.convert("RGB")

    # Apply preprocessing transforms
    transform = get_transform()
    tensor = transform(pil_image)

    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)

    return tensor, pil_image


def validate_image(file):
    """
    Validate that an uploaded file is a valid image.

    Args:
        file: Uploaded file object.

    Returns:
        bool: True if valid, False otherwise.
        str: Error message if invalid, empty string if valid.
    """
    if file is None:
        return False, "No file provided."

    filename = file.filename.lower()
    allowed = {"png", "jpg", "jpeg", "bmp", "tiff"}
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in allowed:
        return False, f"Invalid file type '.{ext}'. Allowed: {', '.join(allowed)}"

    try:
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)  # Reset stream after verify
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
