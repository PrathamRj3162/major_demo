"""
All the config values and constants used throughout the backend.
Paths, model params, preprocessing settings, federated learning defaults etc.
"""

import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# image preprocessing (matching ImageNet stats for DenseNet)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# model settings
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CONFIDENCE_THRESHOLD = 0.70  # anything below this gets flagged for review

# federated learning defaults
NUM_CLIENTS = 4
FED_ROUNDS = 3
LOCAL_EPOCHS = 1
FED_BATCH_SIZE = 16
FED_LEARNING_RATE = 0.001

# allowed image types for upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# chest X-ray validation thresholds (OOD detection)
# if channel diff is above this, image is probably not grayscale (not an X-ray)
GRAYSCALE_THRESHOLD = 25.0
# chest X-rays are roughly square or slightly portrait — reject extreme ratios
ASPECT_RATIO_RANGE = (0.4, 2.5)
# softmax entropy above this means model is very confused (max for 2 classes ≈ 0.693)
SOFTMAX_ENTROPY_THRESHOLD = 0.65
# minimum L2 norm of DenseNet feature vector for a plausible chest X-ray
ACTIVATION_ENERGY_MIN = 5.0

# make sure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
