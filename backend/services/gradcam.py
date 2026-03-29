"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
=====================================================
Implements Grad-CAM for DenseNet121 to provide visual explanations
of the model's predictions. This is critical for medical AI, as
clinicians need to understand WHY the model made a specific prediction.

How Grad-CAM Works:
  1. Perform a forward pass and record activations at the target layer
  2. Compute gradients of the predicted class score w.r.t. those activations
  3. Global-average-pool the gradients to get channel importance weights
  4. Compute weighted sum of activation maps
  5. Apply ReLU to keep only positive influences
  6. Overlay the heatmap on the original image

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from models.model_loader import get_model, get_device
from config import IMAGE_SIZE


class GradCAM:
    """
    Grad-CAM implementation for DenseNet121.
    
    Hooks into the last DenseBlock to capture activations and gradients,
    then generates class-discriminative heatmaps.
    """

    def __init__(self, model=None):
        self.model = model or get_model()
        self.device = get_device()
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        # Target: the last DenseBlock in DenseNet121
        target_layer = self.model.densenet.features.denseblock4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the given image.

        Args:
            image_tensor: Preprocessed tensor (1, 3, 224, 224).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            heatmap: numpy array (224, 224) with values in [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]       # (C, H, W)
        activations = self.activations[0]   # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # Weighted combination of activation maps (vectorized — no Python loop)
        cam = torch.sum(weights[:, None, None] * activations, dim=0)

        # Apply ReLU — only positive contributions
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to image size
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        return cam

    def generate_overlay(self, image_tensor, original_pil_image, target_class=None):
        """
        Generate a Grad-CAM heatmap overlay on the original image.

        Args:
            image_tensor: Preprocessed tensor (1, 3, 224, 224).
            original_pil_image: Original PIL Image (RGB).
            target_class: Class index to explain.

        Returns:
            overlay_image: PIL Image with heatmap overlay.
            heatmap_image: PIL Image of the raw heatmap.
        """
        # Generate heatmap
        heatmap = self.generate(image_tensor, target_class)

        # Resize original image to match heatmap
        original_resized = original_pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        original_np = np.array(original_resized)

        # Convert heatmap to colormap (JET)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay: blend original image with heatmap
        overlay = np.float32(heatmap_colored) * 0.4 + np.float32(original_np) * 0.6
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Convert to PIL Images
        overlay_image = Image.fromarray(overlay)
        heatmap_image = Image.fromarray(heatmap_colored)

        return overlay_image, heatmap_image
