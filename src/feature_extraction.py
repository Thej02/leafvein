"""
src/feature_extraction.py — Feature extraction from segmented leaf images.

Extracts two categories of features:

**From backlit image (vein features):**
  - vein_density: skeleton pixels / leaf area
  - vein_thickness_avg: average vein width via distance transform
  - branch_point_count: proxy for branching complexity

**From front-lit image (color/chlorosis features):**
  - mean_hue, mean_saturation: overall greenness proxies
  - yellow_pixel_ratio: fraction of leaf in the "yellow/pale" HSV band
  - excess_green_index (ExG): 2G − R − B, normalized, averaged over leaf
  - dark_green_color_index (DGCI): standard turf/plant-health formula
  - interveinal_contrast: color difference between on-vein and off-vein regions

Every function is independently testable and produces deterministic output.
No machine learning is used.
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    YELLOW_HSV_LOWER, YELLOW_HSV_UPPER,
    EXG_HEALTHY_LOW, DGCI_HEALTHY_LOW,
    INTERVEINAL_CONTRAST_THRESHOLD, COLOR_SPATIAL_VARIANCE_MAX
)


# ─────────────────────────────────────────────────────────────────────────────
# Vein-based features (from backlit image)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vein_density(vein_pixel_count: int, leaf_area_pixels: int) -> float:
    """
    Compute vein density as the ratio of vein skeleton pixels to leaf area.

    Higher vein density generally indicates healthier, more robust vascular
    architecture. Deficiency may cause reduced vein visibility.

    Args:
        vein_pixel_count: Number of pixels in the vein skeleton.
        leaf_area_pixels: Total number of leaf pixels (from segmentation mask).

    Returns:
        Vein density ratio (0.0 to 1.0, typically 0.01–0.10 for real leaves).
    """
    if leaf_area_pixels == 0:
        return 0.0
    return vein_pixel_count / leaf_area_pixels


def compute_vein_thickness(vein_mask: np.ndarray, skeleton: np.ndarray) -> float:
    """
    Estimate average vein thickness using distance transform.

    The distance transform of the binary vein mask gives, at each vein pixel,
    the distance to the nearest non-vein pixel. Sampling this along the skeleton
    (centerline) gives the half-width of the vein at each point.

    Args:
        vein_mask: Binary vein mask (pre-skeletonization, uint8, 0 or 255).
        skeleton: Binary vein skeleton (uint8, 0 or 255).

    Returns:
        Average vein thickness in pixels (diameter = 2 × distance transform value).
        Returns 0.0 if no skeleton pixels exist.
    """
    if cv2.countNonZero(skeleton) == 0:
        return 0.0

    # Distance transform on the vein mask
    dist_transform = cv2.distanceTransform(vein_mask, cv2.DIST_L2, 5)

    # Sample distances along skeleton pixels
    skeleton_coords = skeleton > 0
    distances = dist_transform[skeleton_coords]

    if len(distances) == 0:
        return 0.0

    # Average thickness = 2 × mean half-width
    return float(2.0 * np.mean(distances))


# ─────────────────────────────────────────────────────────────────────────────
# Color-based features (from front-lit image)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_hue_saturation(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute mean hue and saturation of the leaf region in HSV space.

    Hue indicates color (green vs yellow vs brown), saturation indicates
    color intensity (vivid green vs washed-out pale).

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).

    Returns:
        Dict with 'mean_hue' (0–180) and 'mean_saturation' (0–255).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    leaf_pixels = mask > 0

    if not np.any(leaf_pixels):
        return {'mean_hue': 0.0, 'mean_saturation': 0.0}

    hue_values = hsv[:, :, 0][leaf_pixels]
    sat_values = hsv[:, :, 1][leaf_pixels]

    return {
        'mean_hue': float(np.mean(hue_values)),
        'mean_saturation': float(np.mean(sat_values)),
    }


def compute_yellow_pixel_ratio(image: np.ndarray, mask: np.ndarray,
                                yellow_lower: tuple = YELLOW_HSV_LOWER,
                                yellow_upper: tuple = YELLOW_HSV_UPPER) -> float:
    """
    Compute the fraction of leaf pixels that fall in the "yellow/pale" HSV band.

    This is a direct proxy for chlorosis (yellowing). A healthy leaf should
    have very few yellow pixels; a deficient leaf may have a large fraction.

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).
        yellow_lower: Lower HSV bound for yellow detection.
        yellow_upper: Upper HSV bound for yellow detection.

    Returns:
        Ratio of yellow pixels to total leaf pixels (0.0 to 1.0).
    """
    leaf_area = cv2.countNonZero(mask)
    if leaf_area == 0:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array(yellow_lower), np.array(yellow_upper))

    # Only count yellow pixels within the leaf
    yellow_in_leaf = cv2.bitwise_and(yellow_mask, mask)
    yellow_count = cv2.countNonZero(yellow_in_leaf)

    return yellow_count / leaf_area, yellow_in_leaf


def compute_excess_green_index(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the Excess Green Index (ExG) averaged over leaf pixels.

    ExG = 2G − R − B (using normalized RGB: r = R/(R+G+B), etc.)

    ExG is a well-established vegetation color index in precision agriculture.
    Positive values indicate green vegetation; lower values indicate
    reduced greenness (yellowing, browning).

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).

    Returns:
        Mean ExG across leaf pixels (typically -1.0 to +1.0).
    """
    leaf_pixels = mask > 0
    if not np.any(leaf_pixels):
        return 0.0

    # Extract BGR channels as float
    b = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)

    # Normalize to chromatic coordinates
    total = r + g + b + 1e-10  # avoid division by zero
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total

    # ExG = 2g − r − b (where g, r, b are normalized)
    exg = 2.0 * g_norm - r_norm - b_norm

    # Create binary mask for pixels falling below threshold
    exg_mask = np.zeros_like(mask)
    failing_pixels = (exg < EXG_HEALTHY_LOW) & leaf_pixels
    exg_mask[failing_pixels] = 255

    # Average over leaf pixels only
    return float(np.mean(exg[leaf_pixels])), exg_mask


def compute_dgci(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the Dark Green Color Index (DGCI) averaged over leaf pixels.

    DGCI = [(Hue − 60)/60 + (1 − Saturation) + (1 − Brightness)] / 3

    Where Hue is in degrees (0–360), Saturation and Brightness are 0–1.

    DGCI was developed for turfgrass color assessment and correlates with
    nitrogen status. Higher DGCI = darker green = healthier.

    Reference: Karcher & Richardson (2003), "Quantifying Turfgrass Color
    Using Digital Image Analysis" — Crop Science 43:943–951.

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).

    Returns:
        Mean DGCI across leaf pixels (0.0 to 1.0, higher = darker green).
    """
    leaf_pixels = mask > 0
    if not np.any(leaf_pixels):
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # OpenCV HSV ranges: H=0–180, S=0–255, V=0–255
    # Convert to standard ranges: H=0–360, S=0–1, V=0–1
    h = hsv[:, :, 0].astype(np.float64) * 2.0      # 0–360
    s = hsv[:, :, 1].astype(np.float64) / 255.0     # 0–1
    v = hsv[:, :, 2].astype(np.float64) / 255.0     # 0–1

    # DGCI formula
    hue_component = (h - 60.0) / 60.0
    sat_component = 1.0 - s
    val_component = 1.0 - v

    dgci = (hue_component + sat_component + val_component) / 3.0

    dgci_mask = np.zeros_like(mask)
    failing_pixels = (dgci < DGCI_HEALTHY_LOW) & leaf_pixels
    dgci_mask[failing_pixels] = 255

    return float(np.mean(dgci[leaf_pixels])), dgci_mask


def compute_glare_mask(image: np.ndarray, mask: np.ndarray, v_thresh: int = 220, s_thresh: int = 40) -> np.ndarray:
    """
    Detect specular highlights (glare) on the leaf surface.

    Glare is typically characterized by very high brightness (Value) and very
    low color intensity (Saturation), as it reflects the white light source.
    These regions should be excluded from color variance and contrast calculations.

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).
        v_thresh: Minimum Value (brightness) in HSV (0-255) to be considered glare.
        s_thresh: Maximum Saturation in HSV (0-255) to be considered glare.

    Returns:
        Binary mask (uint8, 0 or 255) of glare pixels inside the leaf.
    """
    if cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    glare = (v_channel > v_thresh) & (s_channel < s_thresh)
    glare_mask = np.zeros_like(mask)
    glare_mask[glare] = 255

    return cv2.bitwise_and(glare_mask, mask)


def compute_interveinal_contrast(image: np.ndarray,
                                   mask: np.ndarray,
                                   skeleton: np.ndarray,
                                   glare_mask: np.ndarray = None,
                                   dilation_radius: int = 10) -> float:
    """
    Compute the color contrast between on-vein and off-vein (interveinal) regions.

    A classic symptom of Mg/Fe deficiency is that veins stay green while the
    tissue between veins yellows. This feature captures that pattern as a
    numeric difference in mean green channel intensity.

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).
        skeleton: Binary vein skeleton from the backlit image analysis (uint8, 0 or 255).
        glare_mask: Optional binary mask of specular highlights to exclude.
        dilation_radius: How many pixels around each skeleton pixel to consider
                        as "on-vein" region.

    Returns:
        Absolute difference in mean green-channel intensity between on-vein
        and off-vein leaf regions. Higher values suggest interveinal chlorosis.
        Returns 0.0 if either region is empty.
    """
    if cv2.countNonZero(skeleton) == 0 or cv2.countNonZero(mask) == 0:
        return 0.0

    # Dilate skeleton to create "on-vein" region (wider than 1 pixel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    vein_region = cv2.dilate(skeleton, kernel, iterations=1)

    # Exclude glare from the base mask
    if glare_mask is not None:
        effective_mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))
    else:
        effective_mask = mask

    # On-vein = vein region ∩ effective leaf mask
    on_vein_mask = cv2.bitwise_and(vein_region, effective_mask)

    # Off-vein = effective leaf mask − on-vein
    off_vein_mask = cv2.bitwise_and(effective_mask, cv2.bitwise_not(vein_region))

    on_vein_pixels = on_vein_mask > 0
    off_vein_pixels = off_vein_mask > 0

    if not np.any(on_vein_pixels) or not np.any(off_vein_pixels):
        return 0.0

    # Compare green channel (index 1 in BGR)
    green = image[:, :, 1].astype(np.float64)

    mean_green_on_vein = np.mean(green[on_vein_pixels])
    mean_green_off_vein = np.mean(green[off_vein_pixels])
    contrast = float(abs(mean_green_on_vein - mean_green_off_vein))

    # Mask of pixels near the vein skeleton whose color deviates from baseline
    contrast_mask = np.zeros_like(mask)
    failing = (np.abs(green - mean_green_on_vein) > INTERVEINAL_CONTRAST_THRESHOLD) & off_vein_pixels
    contrast_mask[failing] = 255

    return contrast, contrast_mask


def compute_color_spatial_variance(image: np.ndarray, mask: np.ndarray, glare_mask: np.ndarray = None) -> float:
    """
    Compute spatial variance of color (hue and value) across the leaf.

    Nutrient chlorosis is usually diffuse and graded. High spatial variance
    suggests sharp, localized patches (e.g., pest damage, fungal spotting,
    or physical injury) rather than a systemic nutrient deficiency. This serves
    as a confound check.

    Args:
        image: Front-lit BGR image.
        mask: Binary leaf mask (uint8, 0 or 255).
        glare_mask: Optional binary mask of specular highlights to exclude.

    Returns:
        Sum of hue variance and value (lightness) variance.
    """
    if glare_mask is not None:
        effective_mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))
    else:
        effective_mask = mask

    leaf_pixels = effective_mask > 0
    if not np.any(leaf_pixels):
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Scale Hue to 0-360 and Value to 0-1 for standard variance
    h = hsv[:, :, 0].astype(np.float64) * 2.0
    v = hsv[:, :, 2].astype(np.float64) / 255.0

    h_leaf = h[leaf_pixels]
    v_leaf = v[leaf_pixels]
    var_h = float(np.var(h_leaf))
    var_v = float(np.var(v_leaf))
    global_variance = var_h + (var_v * 1000)

    # Local variance map via box filter
    k = 21
    mean_h = cv2.boxFilter(h, cv2.CV_64F, (k, k))
    mean_h_sq = cv2.boxFilter(h**2, cv2.CV_64F, (k, k))
    local_var_h = mean_h_sq - mean_h**2

    mean_v = cv2.boxFilter(v, cv2.CV_64F, (k, k))
    mean_v_sq = cv2.boxFilter(v**2, cv2.CV_64F, (k, k))
    local_var_v = mean_v_sq - mean_v**2

    local_variance = local_var_h + (local_var_v * 1000)
    var_mask = np.zeros_like(mask)
    failing = (local_variance > COLOR_SPATIAL_VARIANCE_MAX) & leaf_pixels
    var_mask[failing] = 255

    return global_variance, var_mask

# ─────────────────────────────────────────────────────────────────────────────
# Combined feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(frontlit_image: np.ndarray,
                          mask: np.ndarray,
                          vein_result: dict,
                          leaf_area_pixels: int) -> dict:
    """
    Extract all features from both the backlit (via vein_result) and front-lit images.

    This is the main entry point for feature extraction.

    Args:
        frontlit_image: Preprocessed BGR front-lit leaf image.
        mask: Binary leaf mask (uint8, 0 or 255).
        vein_result: Output dict from vein_extraction.extract_veins(), containing:
                     'skeleton', 'vein_mask', 'vein_pixel_count', 'branch_point_count'.
        leaf_area_pixels: Total leaf area in pixels (from segmentation).

    Returns:
        Dict of all extracted feature values.
    """
    # Vein features (from backlit analysis results)
    vein_density = compute_vein_density(
        vein_result['vein_pixel_count'], leaf_area_pixels
    )
    vein_thickness = compute_vein_thickness(
        vein_result['vein_mask'], vein_result['skeleton']
    )

    # Color features (from front-lit image)
    hue_sat = compute_mean_hue_saturation(frontlit_image, mask)
    yellow_ratio, yellow_mask = compute_yellow_pixel_ratio(frontlit_image, mask)
    exg, exg_mask = compute_excess_green_index(frontlit_image, mask)
    dgci, dgci_mask = compute_dgci(frontlit_image, mask)
    
    # Calculate glare mask for spatial features
    glare_mask = compute_glare_mask(frontlit_image, mask)
    
    interveinal, interveinal_mask = compute_interveinal_contrast(
        frontlit_image, mask, vein_result['skeleton'], glare_mask=glare_mask
    )
    spatial_variance, variance_mask = compute_color_spatial_variance(frontlit_image, mask, glare_mask=glare_mask)

    return {
        # Vein features
        'vein_density': vein_density,
        'vein_thickness_avg': vein_thickness,
        'branch_point_count': vein_result['branch_point_count'],
        'vein_pixel_count': vein_result['vein_pixel_count'],
        'leaf_area_pixels': leaf_area_pixels,

        # Color features (Primary signals)
        'mean_hue': hue_sat['mean_hue'],
        'mean_saturation': hue_sat['mean_saturation'],
        'yellow_pixel_ratio': yellow_ratio,
        'excess_green_index': exg,
        'dgci': dgci,
        'interveinal_contrast': interveinal,
        'color_spatial_variance': spatial_variance,

        # Underlying pixel masks for primary factors (used in Stage 10 circling)
        'masks': {
            'yellow_pixel_ratio': yellow_mask,
            'excess_green_index': exg_mask,
            'dgci': dgci_mask,
            'interveinal_contrast': interveinal_mask,
            'color_spatial_variance': variance_mask,
        }
    }
