"""
severityClass/rule_based.py — Rule-based Rosa-sinensis leaf health severity scoring.

Reuses existing vein extraction and color feature extraction modules by importing them.
Calculates a continuous Severity Index (0–100) combining:
  1. Interveinal yellowing extent (yellow pixel ratio)
  2. Vein-to-interveinal green contrast (interveinal chlorosis)
  3. Vein density deviation from healthy baseline
  4. Dark Green Color Index (DGCI) loss
  5. High spatial variance penalty (pest/confound check)

Buckets into 4 classes: Healthy, Mild, Moderate, Severe.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import cv2
import numpy as np

# Ensure project root is on path for importing existing modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_image,
    resize_to_working_resolution,
    denoise,
    normalize_brightness,
)
from src.interactive_roi import select_circle_roi
from src.segmentation import segment_leaf
from src.species_check import verify_hibiscus_species
from src.vein_extraction import extract_veins
from src.feature_extraction import extract_all_features
from config.thresholds import (
    VEIN_DENSITY_DEFICIENT,
    INTERVEINAL_CONTRAST_THRESHOLD,
    YELLOW_RATIO_DEFICIENT,
    DGCI_HEALTHY_LOW,
    COLOR_SPATIAL_VARIANCE_MAX,
)
from severityClass.recommendations import get_recommendation


# ─────────────────────────────────────────────────────────────────────────────
# Threshold definitions & weights
# ─────────────────────────────────────────────────────────────────────────────
HEALTHY_VEIN_DENSITY_BASELINE = 0.0550  # Calibrated healthy reference mean
HEALTHY_DGCI_BASELINE = 0.6500          # Deep green threshold
BASELINE_CONTRAST_FLOOR = 25.0          # Below this, vein-to-lamina contrast is uniform

# Weights for multi-criteria severity index (sum to 1.0)
WEIGHT_YELLOWING = 0.35
WEIGHT_CONTRAST = 0.25
WEIGHT_VEIN_DENSITY = 0.20
WEIGHT_DGCI = 0.20

# Class score bounds (0 to 100)
THRESHOLD_HEALTHY_MAX = 20.0    # 0.0 - 19.9 -> Healthy
THRESHOLD_MILD_MAX = 45.0       # 20.0 - 44.9 -> Mild
THRESHOLD_MODERATE_MAX = 70.0   # 45.0 - 69.9 -> Moderate
                                # >= 70.0 -> Severe


def compute_severity(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a continuous Severity Score (0 to 100) and classify into:
    'Healthy', 'Mild', 'Moderate', or 'Severe'.

    Args:
        features: Dictionary containing extracted botanical features from feature_extraction.py.
                  Required keys: 'yellow_pixel_ratio', 'interveinal_contrast',
                                 'vein_density', 'dgci', 'color_spatial_variance'.

    Returns:
        Dictionary with:
          'severity_score': float (0.0 to 100.0)
          'severity_level': str ('Healthy', 'Mild', 'Moderate', 'Severe')
          'components': dict of individual component scores (0-100)
          'explanation': plain language reasoning for the classification
          'recommendation': dictionary of actionable horticultural guidance
    """
    yellow_ratio = float(features.get('yellow_pixel_ratio', 0.0))
    contrast = float(features.get('interveinal_contrast', 0.0))
    vein_density = float(features.get('vein_density', 0.0))
    dgci = float(features.get('dgci', 0.70))
    variance = float(features.get('color_spatial_variance', 0.0))

    # 1. Yellowing Extent Score (0 to 100):
    # Saturates at 60% yellow pixels (which represents total chlorotic collapse)
    s_yellow = min(100.0, max(0.0, (yellow_ratio / 0.60) * 100.0))

    # 2. Vein-to-Interveinal Contrast Score (0 to 100):
    # Baseline green is <= 25.0. Severe interveinal chlorosis reaches 65.0+.
    if contrast <= BASELINE_CONTRAST_FLOOR:
        s_contrast = 0.0
    else:
        s_contrast = min(100.0, max(0.0, ((contrast - BASELINE_CONTRAST_FLOOR) / 40.0) * 100.0))

    # 3. Vein Density Deviation Score (0 to 100):
    # Measures vascular shortfall relative to healthy baseline
    if vein_density >= HEALTHY_VEIN_DENSITY_BASELINE:
        s_density = 0.0
    else:
        density_deficit = (HEALTHY_VEIN_DENSITY_BASELINE - vein_density) / HEALTHY_VEIN_DENSITY_BASELINE
        s_density = min(100.0, max(0.0, density_deficit * 100.0))

    # 4. DGCI Loss Score (0 to 100):
    # DGCI drops from 0.70 (deep green) down to 0.30 (bleached pale)
    if dgci >= HEALTHY_DGCI_BASELINE:
        s_dgci = 0.0
    else:
        dgci_deficit = (HEALTHY_DGCI_BASELINE - dgci) / 0.35
        s_dgci = min(100.0, max(0.0, dgci_deficit * 100.0))

    # 5. Confound penalty: Sharp patchy variance (pest/injury mottling)
    confound_penalty = 0.0
    if variance > COLOR_SPATIAL_VARIANCE_MAX:
        # Scale penalty from 0 up to 10 points
        excess_var = min(500.0, variance - COLOR_SPATIAL_VARIANCE_MAX)
        confound_penalty = (excess_var / 500.0) * 10.0

    # Composite weighted severity score
    base_score = (
        (WEIGHT_YELLOWING * s_yellow) +
        (WEIGHT_CONTRAST * s_contrast) +
        (WEIGHT_VEIN_DENSITY * s_density) +
        (WEIGHT_DGCI * s_dgci)
    )
    final_score = min(100.0, max(0.0, base_score + confound_penalty))

    # Severity classification bucketing
    if final_score < THRESHOLD_HEALTHY_MAX:
        severity_level = "Healthy"
    elif final_score < THRESHOLD_MILD_MAX:
        severity_level = "Mild"
    elif final_score < THRESHOLD_MODERATE_MAX:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    # Plain language explanation
    explanation_parts = []
    if severity_level == "Healthy":
        explanation_parts.append(
            f"Severity Score is {final_score:.1f}/100 (<{THRESHOLD_HEALTHY_MAX:.0f}), indicating optimal vitality. "
            f"Yellow pixel ratio is low ({yellow_ratio:.1%}), greenness is strong (DGCI {dgci:.2f}), "
            f"and vein density is healthy ({vein_density:.4f})."
        )
    else:
        explanation_parts.append(
            f"Severity Score is {final_score:.1f}/100, classified as {severity_level.upper()} deficiency/stress."
        )
        if s_yellow > 30.0:
            explanation_parts.append(f"Significant chlorosis detected: {yellow_ratio:.1%} of leaf lamina is yellowed.")
        if s_contrast > 30.0:
            explanation_parts.append(f"Interveinal contrast is elevated ({contrast:.1f}), characteristic of Mg/Fe deficiency.")
        if s_density > 25.0:
            explanation_parts.append(f"Vein density is reduced ({vein_density:.4f} vs baseline {HEALTHY_VEIN_DENSITY_BASELINE:.4f}).")
        if s_dgci > 25.0:
            explanation_parts.append(f"Dark Green Color Index is depleted ({dgci:.2f} vs healthy {HEALTHY_DGCI_BASELINE:.2f}).")
        if confound_penalty > 0.0:
            explanation_parts.append(f"High spatial color variance ({variance:.1f}) indicates sharp localized spotting/pest damage.")

    explanation = " ".join(explanation_parts)
    recommendation = get_recommendation(severity_level)

    return {
        "severity_score": round(final_score, 2),
        "severity_level": severity_level,
        "components": {
            "yellowing_component": round(s_yellow, 2),
            "interveinal_contrast_component": round(s_contrast, 2),
            "vein_density_deficit_component": round(s_density, 2),
            "dgci_deficit_component": round(s_dgci, 2),
            "confound_penalty": round(confound_penalty, 2),
        },
        "thresholds_used": {
            "healthy": f"< {THRESHOLD_HEALTHY_MAX:.0f}",
            "mild": f"{THRESHOLD_HEALTHY_MAX:.0f} - {THRESHOLD_MILD_MAX:.0f}",
            "moderate": f"{THRESHOLD_MILD_MAX:.0f} - {THRESHOLD_MODERATE_MAX:.0f}",
            "severe": f">= {THRESHOLD_MODERATE_MAX:.0f}",
        },
        "explanation": explanation,
        "recommendation": recommendation,
    }


def analyze_image_severity(image_path: str,
                           backlit_path: str = None,
                           output_dir: str = None,
                           use_interactive_roi: bool = False,
                           display_overlay: bool = True) -> Dict[str, Any]:
    """
    Run leaf health feature extraction and evaluate rule-based severity for a given leaf image.

    Args:
        image_path: Path to front-lit (or single) leaf image.
        backlit_path: Optional path to backlit image (if paired mode). Defaults to image_path.
        output_dir: Optional directory to save annotated result visualization.
        use_interactive_roi: If True, opens interactive window to draw ROI polygon.
        display_overlay: If True, displays vein overlay window for visual confirmation.

    Returns:
        Complete evaluation result dict including features, severity score, and recommendation.
    """
    if backlit_path is None:
        backlit_path = image_path

    # Step 1: Preprocessing
    print("[1/5] Loading and preprocessing images...")
    raw_front = load_image(image_path)
    raw_back = load_image(backlit_path)

    resized_front = resize_to_working_resolution(raw_front)
    resized_back = resize_to_working_resolution(raw_back)

    denoised_front = denoise(resized_front)
    denoised_back = denoise(resized_back)

    prep_front = normalize_brightness(denoised_front)
    prep_back = normalize_brightness(denoised_back)

    # Step 2: Segmentation & Species Check
    print("[2/5] Segmenting leaf and verifying species...")
    seg_res = segment_leaf(prep_front)
    mask = seg_res['mask']

    species_res = verify_hibiscus_species(mask)
    if not species_res['is_hibiscus']:
        return {
            'error': f"Species check rejected: {species_res['reason']}",
            'is_hibiscus': False,
            'image_path': image_path
        }
    print("      Species Check: Verified Hibiscus (Rosa-sinensis).")

    # Interactive ROI Tracing ("the ROI and all")
    if use_interactive_roi:
        print("       Opening interactive ROI selection window...")
        roi_mask = select_circle_roi(resized_front)
        mask = cv2.bitwise_and(mask, roi_mask)

    leaf_area = cv2.countNonZero(mask)
    print(f"      Effective Leaf Area: {leaf_area:,} pixels")

    # Step 3: Vein extraction
    print("[3/5] Extracting vein architecture...")
    vein_res = extract_veins(prep_back, mask)

    # Display vein overlay confirmation window
    should_display = display_overlay and os.environ.get('HEADLESS', '').lower() not in ('1', 'true', 'yes')
    if should_display:
        print("       Displaying vein overlay for visual confirmation (Press any key to proceed)...")
        window_name = "Vein Skeleton Confirmation (Press any key to continue)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, vein_res['debug_overlay'])
        cv2.waitKey(0)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    # Step 4: Feature extraction from the ROI-intersected mask
    print("[4/5] Extracting botanical and chlorosis features...")
    features = extract_all_features(prep_front, mask, vein_res, leaf_area)

    # Step 5: Severity Scoring
    print("[5/5] Evaluating leaf health severity...")
    severity_eval = compute_severity(features)

    # Step 6: Generate annotated visualization if requested
    annotated_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(output_dir, f"{img_id}_severity_annotated.jpg")
        
        # Create visual annotation overlay
        overlay = vein_res['debug_overlay'].copy()
        
        # Color mapping for severity banner
        banner_colors = {
            "Healthy": (50, 180, 50),     # Green
            "Mild": (50, 200, 240),       # Amber / Yellow
            "Moderate": (20, 130, 240),   # Orange
            "Severe": (40, 40, 220),      # Red
        }
        color = banner_colors.get(severity_eval['severity_level'], (128, 128, 128))
        
        # Header banner
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, 80), (w, 85), color, -1)
        
        title = f"SEVERITY: {severity_eval['severity_level'].upper()} (Score: {severity_eval['severity_score']:.1f}/100)"
        cv2.putText(overlay, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        metrics_str = f"Yellow: {features['yellow_pixel_ratio']:.1%} | DGCI: {features['dgci']:.2f} | Contrast: {features['interveinal_contrast']:.1f} | Density: {features['vein_density']:.4f}"
        cv2.putText(overlay, metrics_str, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        
        cv2.imwrite(annotated_path, overlay)
    return {
        'image_path': image_path,
        'is_hibiscus': True,
        'features': features,
        'severity': severity_eval,
        'annotated_path': annotated_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-Based Rosa-sinensis Leaf Health Severity Classifier")
    parser.add_argument('--image', type=str, required=True, help="Path to leaf image")
    parser.add_argument('--backlit', type=str, default=None, help="Optional separate backlit image")
    parser.add_argument('--output', '-o', type=str, default=None, help="Output directory for annotated image")
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument('--select-roi', dest='select_roi', action='store_true', default=True,
                           help="Interactively select ROI with lasso polygon (default: True)")
    roi_group.add_argument('--no-roi', dest='select_roi', action='store_false',
                           help="Skip interactive ROI selection and use full segmented leaf")
    parser.add_argument('--no-display', action='store_true', help="Suppress interactive GUI popup window for vein overlay confirmation")
    args = parser.parse_args()

    print("=" * 70)
    print("  Rosa-sinensis Leaf Health Severity Evaluator (Rule-Based)")
    print("=" * 70)

    is_headless = os.environ.get('HEADLESS', '').lower() in ('1', 'true', 'yes')
    use_roi = args.select_roi and not args.no_display and not is_headless
    display_overlay = not args.no_display and not is_headless

    result = analyze_image_severity(
        args.image,
        args.backlit,
        args.output,
        use_interactive_roi=use_roi,
        display_overlay=display_overlay
    )
    if 'error' in result:
        print(f"\nX Error: {result['error']}")
        sys.exit(1)

    sev = result['severity']
    print(f"\nImage:            {args.image}")
    print(f"Severity Score:   {sev['severity_score']:.1f} / 100")
    print(f"Severity Class:   {sev['severity_level'].upper()}")
    print("-" * 70)
    print(f"Component Breakdown:")
    for k, v in sev['components'].items():
        print(f"  - {k.replace('_', ' ').title()}: {v:.1f}")
    print("-" * 70)
    print(f"Explanation:\n  {sev['explanation']}")
    print("-" * 70)
    rec = sev['recommendation']
    print(f"Actionable Recommendation ({rec['status']}):")
    print(f"  Focus:   {rec['nutrient_focus']}")
    print(f"  Urgency: {rec['urgency']}")
    print("  Steps to take:")
    for step in rec['actions']:
        print(f"    * {step}")
    if result.get('annotated_path'):
        print(f"\nSaved annotated visualization: {result['annotated_path']}")


if __name__ == '__main__':
    main()
