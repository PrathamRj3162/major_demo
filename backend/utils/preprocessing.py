"""
Image preprocessing pipeline for the DenseNet model.
Handles loading images from different sources (file path, bytes, PIL),
resizing to 224x224, converting to RGB, normalising with ImageNet
stats, and adding the batch dimension the model expects.
"""

import io
import numpy as np
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transform():
    """Standard transform chain — resize, tensor, normalise."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image_source):
    """
    Take an image (path, bytes, or PIL object), run it through the
    preprocessing pipeline, and return the tensor + original PIL image.
    The PIL image is kept around because Grad-CAM needs it for the overlay.
    """
    # figure out what we got and open it
    if isinstance(image_source, str):
        pil_image = Image.open(image_source)
    elif isinstance(image_source, bytes):
        pil_image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, Image.Image):
        pil_image = image_source
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    # X-rays are often grayscale, model needs RGB
    pil_image = pil_image.convert("RGB")

    transform = get_transform()
    tensor = transform(pil_image)

    # model expects shape (batch, channels, h, w)
    tensor = tensor.unsqueeze(0)

    return tensor, pil_image


def validate_image(file):
    """
    Quick check that an uploaded file is actually a valid image
    in one of our supported formats.
    Returns (True, "") if ok, or (False, error_message) if not.
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
        file.stream.seek(0)  # reset so Flask can read it again later
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
