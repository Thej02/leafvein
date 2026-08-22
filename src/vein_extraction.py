"""
src/vein_extraction.py — Leaf vein extraction from backlit images.

Takes a backlit leaf image (with leaf mask from segmentation) and produces:
  1. A binary vein skeleton (1-pixel-wide vein network)
  2. Vein pixel count
  3. Branch point count (proxy for branching complexity)
  4. Debug overlay image for visual QA

The approach uses classical image processing:
  - CLAHE for vein contrast enhancement
  - Frangi vesselness filter (designed for thin curvilinear structures)
  - Adaptive thresholding as alternative/complement
  - Skeletonization via scikit-image
  - Small-branch pruning to reduce noise

No machine learning is used at any stage.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    VEIN_CLAHE_CLIP_LIMIT,
    VEIN_CLAHE_TILE_GRID_SIZE,
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    FRANGI_SIGMAS,
    FRANGI_BLACK_RIDGES,
    MIN_SKELETON_BRANCH_LENGTH,
)


def enhance_vein_contrast(gray: np.ndarray,
                           clip_limit: float = VEIN_CLAHE_CLIP_LIMIT,
                           tile_grid_size: tuple = VEIN_CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """
    Enhance vein visibility in a grayscale backlit leaf image using CLAHE.

    The backlit image typically shows veins as darker lines against brighter
    translucent tissue. CLAHE amplifies this local contrast.

    Args:
        gray: Grayscale image of the backlit leaf (masked to leaf region).
        clip_limit: CLAHE contrast clipping limit (higher = more aggressive).
        tile_grid_size: Grid size for CLAHE tiles.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def extract_veins_adaptive(gray_enhanced: np.ndarray,
                            mask: np.ndarray,
                            block_size: int = ADAPTIVE_THRESH_BLOCK_SIZE,
                            c: int = ADAPTIVE_THRESH_C) -> np.ndarray:
    """
    Extract veins using adaptive thresholding.

    Adaptive thresholding works well when veins are locally darker/lighter
    than their immediate surroundings. The method adapts to local brightness
    variations, which is useful for uneven backlighting.

    Args:
        gray_enhanced: CLAHE-enhanced grayscale leaf image.
        mask: Binary leaf mask (0 or 255).
        block_size: Neighborhood size for adaptive threshold (must be odd).
        c: Constant subtracted from the mean.

    Returns:
        Binary vein mask (uint8, 0 or 255).
    """
    # Adaptive threshold — veins appear as locally darker structures in backlit
    binary = cv2.adaptiveThreshold(
        gray_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c
    )

    # Apply leaf mask to remove background artifacts
    binary = cv2.bitwise_and(binary, mask)

    return binary


def extract_veins_frangi(gray_enhanced: np.ndarray,
                          mask: np.ndarray,
                          sigmas=FRANGI_SIGMAS,
                          black_ridges: bool = FRANGI_BLACK_RIDGES) -> np.ndarray:
    """
    Extract veins using the Frangi vesselness filter.

    The Frangi filter is specifically designed for detecting thin curvilinear
    structures (originally blood vessels in medical imaging). It computes the
    eigenvalues of the Hessian matrix at multiple scales and returns a
    "vesselness" response that peaks on elongated ridge structures like veins.

    This is classical differential geometry, not machine learning.

    Args:
        gray_enhanced: CLAHE-enhanced grayscale leaf image.
        mask: Binary leaf mask (0 or 255).
        sigmas: Range of scales (vein widths in pixels) to detect.
        black_ridges: If True, detect dark ridges on bright background.

    Returns:
        Binary vein mask (uint8, 0 or 255).
    """
    # Normalize to [0, 1] float for scikit-image
    gray_float = gray_enhanced.astype(np.float64) / 255.0

    # Apply Frangi filter
    vesselness = frangi(gray_float, sigmas=sigmas, black_ridges=black_ridges)

    # Normalize vesselness to [0, 255] and threshold
    if vesselness.max() > 0:
        vesselness_norm = (vesselness / vesselness.max() * 255).astype(np.uint8)
    else:
        vesselness_norm = np.zeros_like(gray_enhanced, dtype=np.uint8)

    # Otsu threshold on the vesselness response
    _, binary = cv2.threshold(vesselness_norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply leaf mask
    binary = cv2.bitwise_and(binary, mask)

    return binary


def combine_vein_detections(adaptive_veins: np.ndarray,
                             frangi_veins: np.ndarray) -> np.ndarray:
    """
    Combine vein detections from adaptive thresholding and Frangi filter.

    Uses the intersection (AND) of both methods to reduce false positives —
    a pixel is considered a vein only if both methods agree.

    Args:
        adaptive_veins: Binary vein mask from adaptive thresholding.
        frangi_veins: Binary vein mask from Frangi filter.

    Returns:
        Combined binary vein mask.
    """
    # Use union (OR) to capture veins detected by either method.
    # If this produces too many false positives, switch to intersection (AND).
    combined = cv2.bitwise_or(adaptive_veins, frangi_veins)

    # Clean up with morphological closing to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return combined


def skeletonize_veins(vein_mask: np.ndarray) -> np.ndarray:
    """
    Reduce the vein mask to a 1-pixel-wide skeleton.

    Uses the scikit-image skeletonize function (Lee 1994 algorithm), which
    is a classical morphological thinning operation.

    Args:
        vein_mask: Binary vein mask (uint8, 0 or 255).

    Returns:
        Skeleton image (uint8, 0 or 255) — 1-pixel-wide vein network.
    """
    # Convert to boolean for scikit-image
    vein_bool = vein_mask > 0

    # Remove very small objects before skeletonizing (noise)
    vein_clean = remove_small_objects(vein_bool,
                                      min_size=MIN_SKELETON_BRANCH_LENGTH,
                                      connectivity=2)

    # Skeletonize
    skeleton_bool = skeletonize(vein_clean)

    # Convert back to uint8
    skeleton = (skeleton_bool.astype(np.uint8)) * 255

    return skeleton


def count_branch_points(skeleton: np.ndarray) -> int:
    """
    Count branch points (nodes with ≥3 neighbors) in a skeleton image.

    A branch point is where three or more skeleton paths converge — a proxy
    for branching complexity of the vein network.

    Args:
        skeleton: Binary skeleton image (uint8, 0 or 255).

    Returns:
        Number of branch points.
    """
    skel_bool = (skeleton > 0).astype(np.uint8)

    # Convolve with a 3×3 all-ones kernel to count neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_bool, -1, kernel)

    # A branch point has the pixel itself (1) + ≥3 neighbors, so total ≥ 4
    branch_points = np.logical_and(skel_bool > 0, neighbor_count >= 4)

    return int(np.sum(branch_points))


def create_debug_overlay(original_image: np.ndarray,
                          skeleton: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
    """
    Create a visual overlay of the vein skeleton on the original image.

    The skeleton is drawn in red (BGR: 0, 0, 255) over the leaf, with
    non-leaf areas dimmed. This is used for visual QA — mandatory before
    trusting any downstream numbers.

    Args:
        original_image: Original BGR leaf image (preprocessed).
        skeleton: Binary vein skeleton (uint8, 0 or 255).
        mask: Binary leaf mask (uint8, 0 or 255).

    Returns:
        BGR overlay image.
    """
    overlay = original_image.copy()

    # Dim non-leaf areas
    non_leaf = mask == 0
    overlay[non_leaf] = (overlay[non_leaf] * 0.3).astype(np.uint8)

    # Draw skeleton in red
    skeleton_pixels = skeleton > 0

    # Dilate skeleton slightly for visibility
    skeleton_dilated = cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)
    visible_pixels = skeleton_dilated > 0

    overlay[visible_pixels] = [0, 0, 255]  # Red in BGR

    return overlay


def extract_veins(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Full vein extraction pipeline: enhance → detect → skeletonize.

    This is the main entry point for vein extraction.

    Args:
        image: Preprocessed BGR backlit leaf image.
        mask: Binary leaf mask from segmentation (uint8, 0 or 255).

    Returns:
        Dict with keys:
          'vein_mask': binary vein mask before skeletonization
          'skeleton': 1-pixel-wide vein skeleton (uint8)
          'vein_pixel_count': int — number of skeleton pixels
          'branch_point_count': int — number of branch points in skeleton
          'debug_overlay': BGR image with skeleton drawn in red over leaf
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply mask to focus on leaf region only
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Enhance contrast
    gray_enhanced = enhance_vein_contrast(gray_masked)

    # Extract veins via both methods
    adaptive_veins = extract_veins_adaptive(gray_enhanced, mask)
    frangi_veins = extract_veins_frangi(gray_enhanced, mask)

    # Combine
    vein_mask = combine_vein_detections(adaptive_veins, frangi_veins)

    # Skeletonize
    skeleton = skeletonize_veins(vein_mask)

    # Count metrics
    vein_pixel_count = int(cv2.countNonZero(skeleton))
    branch_points = count_branch_points(skeleton)

    # Debug overlay
    debug_overlay = create_debug_overlay(image, skeleton, mask)

    return {
        'vein_mask': vein_mask,
        'skeleton': skeleton,
        'vein_pixel_count': vein_pixel_count,
        'branch_point_count': branch_points,
        'debug_overlay': debug_overlay,
    }
