"""
Configuration module for the AI Diagnostics System.
Contains all constants, thresholds, and paths used across the application.
"""

import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# ─── Image Preprocessing ─────────────────────────────────
IMAGE_SIZE = 224  # DenseNet121 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Model ────────────────────────────────────────────────
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CONFIDENCE_THRESHOLD = 0.70  # Below this → "Needs Review"

# ─── Federated Learning ──────────────────────────────────
NUM_CLIENTS = 4
FED_ROUNDS = 5
LOCAL_EPOCHS = 2
FED_BATCH_SIZE = 16
FED_LEARNING_RATE = 0.001

# ─── Allowed Upload Extensions ────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# ─── Ensure upload directory exists ───────────────────────
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
