"""
src/segmentation.py — Leaf segmentation from background.

Takes a preprocessed image and produces:
  1. A binary mask of the leaf region
  2. A masked leaf image (background pixels set to black)
  3. Leaf area in pixels

The segmentation uses HSV thresholding to isolate green-ish regions,
morphological cleanup, and largest-connected-component selection.
All tunable parameters come from config/thresholds.py.
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    LEAF_HSV_LOWER,
    LEAF_HSV_UPPER,
    MIN_LEAF_AREA_FRACTION,
    MORPH_KERNEL_SIZE,
)


def create_leaf_mask(image: np.ndarray,
                     hsv_lower: tuple = LEAF_HSV_LOWER,
                     hsv_upper: tuple = LEAF_HSV_UPPER,
                     morph_kernel_size: int = MORPH_KERNEL_SIZE) -> np.ndarray:
    """
    Create a binary mask isolating the leaf from the background.

    Steps:
      1. Convert to HSV color space
      2. Threshold on hue/saturation to capture green-ish leaf pixels
      3. Morphological opening (remove small noise) then closing (fill small gaps)
      4. Select the largest connected component as the leaf region

    Args:
        image: Preprocessed BGR image.
        hsv_lower: Lower HSV bound for leaf detection (H, S, V).
        hsv_upper: Upper HSV bound for leaf detection (H, S, V).
        morph_kernel_size: Kernel size for morphological operations.

    Returns:
        Binary mask (uint8, 0 or 255) where 255 = leaf pixel.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Initial threshold
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Select largest connected component
    mask = _keep_largest_component(mask)

    return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    This removes small noise blobs that survived morphological cleanup,
    ensuring the mask represents a single leaf region.

    Args:
        mask: Binary mask (uint8, 0 or 255).

    Returns:
        Cleaned binary mask with only the largest component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        # No foreground components found
        return mask

    # Label 0 is the background — find the largest non-background component
    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
    largest_label = 1 + np.argmax(areas)

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255

    return clean_mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an image, setting non-leaf pixels to black.

    Args:
        image: BGR image.
        mask: Binary mask (uint8, 0 or 255).

    Returns:
        Masked BGR image (background pixels = [0, 0, 0]).
    """
    return cv2.bitwise_and(image, image, mask=mask)


def compute_leaf_area(mask: np.ndarray) -> int:
    """
    Count the number of leaf pixels in the mask.

    Args:
        mask: Binary mask (uint8, 0 or 255).

    Returns:
        Number of non-zero (leaf) pixels.
    """
    return int(cv2.countNonZero(mask))


def validate_segmentation(mask: np.ndarray, image_shape: tuple,
                           min_area_fraction: float = MIN_LEAF_AREA_FRACTION) -> dict:
    """
    Validate that the segmentation result is reasonable.

    Checks:
      - Leaf area is above the minimum fraction of the image
      - The mask has exactly one connected component

    Args:
        mask: Binary mask (uint8, 0 or 255).
        image_shape: Shape of the original image (H, W, C).
        min_area_fraction: Minimum acceptable leaf area as fraction of image area.

    Returns:
        Dict with keys:
          'valid': bool — overall pass/fail
          'leaf_area_pixels': int
          'leaf_area_fraction': float
          'num_components': int
          'issues': list of str — human-readable issue descriptions
    """
    total_pixels = image_shape[0] * image_shape[1]
    leaf_area = compute_leaf_area(mask)
    leaf_fraction = leaf_area / total_pixels if total_pixels > 0 else 0.0

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_components = max(0, num_labels - 1)  # subtract background label

    issues = []

    if leaf_fraction < min_area_fraction:
        issues.append(
            f"Leaf area ({leaf_fraction:.1%}) is below minimum threshold "
            f"({min_area_fraction:.1%}). The leaf may be too small in the frame, "
            f"or segmentation may have failed."
        )

    if num_components == 0:
        issues.append("No leaf region detected. Check lighting and background contrast.")
    elif num_components > 1:
        issues.append(
            f"Multiple connected components ({num_components}) detected after cleanup. "
            f"Expected exactly 1. This may indicate fragmented segmentation."
        )

    return {
        'valid': len(issues) == 0,
        'leaf_area_pixels': leaf_area,
        'leaf_area_fraction': leaf_fraction,
        'num_components': num_components,
        'issues': issues,
    }


def segment_leaf(image: np.ndarray) -> dict:
    """
    Full segmentation pipeline: threshold → cleanup → validate.

    This is the main entry point for segmentation.

    Args:
        image: Preprocessed BGR image.

    Returns:
        Dict with keys:
          'mask': binary leaf mask (uint8)
          'masked_image': BGR image with background removed
          'leaf_area_pixels': int — count of leaf pixels
          'validation': dict — output of validate_segmentation()
    """
    mask = create_leaf_mask(image)
    masked_image = apply_mask(image, mask)
    leaf_area = compute_leaf_area(mask)
    validation = validate_segmentation(mask, image.shape)

    return {
        'mask': mask,
        'masked_image': masked_image,
        'leaf_area_pixels': leaf_area,
        'validation': validation,
    }
