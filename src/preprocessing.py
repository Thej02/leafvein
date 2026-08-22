"""
src/preprocessing.py — Image preprocessing for the leaf health detection pipeline.

Takes a raw captured image and produces a cleaned, normalized version suitable
for downstream segmentation and feature extraction.

Operations:
  1. Resize to standard working resolution (longest edge = WORKING_RESOLUTION)
  2. Light denoising (preserve fine vein detail)
  3. Brightness/contrast normalization via CLAHE on the L channel in LAB space
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path for config imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    WORKING_RESOLUTION,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_H,
    DENOISE_H_COLOR,
)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk in BGR color space (OpenCV default).

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        BGR image as numpy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file exists but cannot be read as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image (corrupt or unsupported format): {image_path}")

    return img


def resize_to_working_resolution(image: np.ndarray,
                                  max_edge: int = WORKING_RESOLUTION) -> np.ndarray:
    """
    Resize image so its longest edge equals `max_edge`, preserving aspect ratio.

    This ensures consistent downstream measurements regardless of the camera's
    native resolution.

    Args:
        image: Input BGR image.
        max_edge: Target size for the longest edge in pixels.

    Returns:
        Resized BGR image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image.copy()

    scale = max_edge / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray,
            h: int = DENOISE_H,
            h_color: int = DENOISE_H_COLOR) -> np.ndarray:
    """
    Apply light denoising while preserving fine detail (e.g., leaf veins).

    Uses cv2.fastNlMeansDenoisingColored — a non-local means denoiser that
    works in LAB space internally and is good at preserving edges.

    Args:
        image: Input BGR image.
        h: Filter strength for luminance component.
        h_color: Filter strength for color components.

    Returns:
        Denoised BGR image.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h_color, 7, 21)


def normalize_brightness(image: np.ndarray,
                          clip_limit: float = CLAHE_CLIP_LIMIT,
                          tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """
    Normalize brightness and contrast using CLAHE on the L channel in LAB space.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local
    contrast while preventing over-amplification of noise, making it ideal for
    images with uneven illumination (common with DIY backlighting setups).

    Args:
        image: Input BGR image.
        clip_limit: CLAHE contrast clipping limit.
        tile_grid_size: Size of the grid for CLAHE processing.

    Returns:
        Brightness-normalized BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_normalized = clahe.apply(l_channel)

    lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline: resize → denoise → normalize brightness.

    Args:
        image: Raw input BGR image.

    Returns:
        Preprocessed BGR image at standard working resolution.
    """
    resized = resize_to_working_resolution(image)
    denoised = denoise(resized)
    normalized = normalize_brightness(denoised)
    return normalized


def preprocess_from_path(image_path: str) -> np.ndarray:
    """
    Convenience function: load an image from disk and preprocess it.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed BGR image.
    """
    raw = load_image(image_path)
    return preprocess(raw)
