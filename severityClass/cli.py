"""
severityClass/cli.py — Command-Line Interface for Rosa-sinensis Leaf Health & Severity Classification.

Executes the complete end-to-end diagnosis pipeline:
  1. Preprocessing & Brightness Normalization
  2. Hibiscus Species Verification (Hard Gate)
  3. Interactive Freehand ROI Lasso Selection (select_circle_roi)
  4. Vein Architecture Extraction & Vein Skeleton Confirmation Window
  5. Multi-Feature Botanical & Chlorosis Extraction on ROI Mask
  6. Dual Verdict:
     - Original Binary / 3-class Diagnostic Status (Healthy / Deficient)
     - 4-Class Continuous Severity Grading (Healthy / Mild / Moderate / Severe)
  7. Cross-Model Benchmarking (Rule-Based, Random Forest ML, MobileNetV2 DL)
  8. Actionable Botanical Guidance (Nutrient targets, Soil pH adjustments, Urgency)

Usage:
  # Single image with interactive ROI lasso & vein overlay confirmation:
  python -m severityClass.cli --image sample_images/1.jpeg

  # Paired image mode (backlit + frontlit):
  python -m severityClass.cli --backlit data/raw/1_backlit.jpeg --frontlit data/raw/1_frontlit.jpeg

  # Headless or non-interactive (automated testing / batch):
  python -m severityClass.cli --image sample_images/1.jpeg --no-roi --no-display

  # Save annotated reports and images:
  python -m severityClass.cli --image sample_images/1.jpeg --output results/
"""

import argparse
import os
import sys
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

from src.preprocessing import (
    load_image,
    resize_to_working_resolution,
    denoise,
    normalize_brightness,
)
from src.segmentation import segment_leaf
from src.species_check import verify_hibiscus_species
from src.interactive_roi import select_circle_roi
from src.vein_extraction import extract_veins
from src.feature_extraction import extract_all_features
from src.decision_engine import evaluate
from severityClass.rule_based import compute_severity
from severityClass.recommendations import get_recommendation


def run_severity_pipeline(
    image_path: str = None,
    backlit_path: str = None,
    frontlit_path: str = None,
    output_dir: str = None,
    use_interactive_roi: bool = True,
    display_overlay: bool = True,
    include_ml: bool = True,
    include_dl: bool = True,
) -> Dict[str, Any]:
    """
    Run complete leaf diagnosis and severity classification pipeline.
    """
    # Determine front and back paths
    if backlit_path and frontlit_path:
        f_path = frontlit_path
        b_path = backlit_path
        mode_str = "Paired Frontlit + Backlit"
    elif image_path:
        f_path = image_path
        b_path = image_path
        mode_str = "Single Image (Front & Transmitted proxy)"
    else:
        raise ValueError("Must provide either --image or both --backlit and --frontlit.")

    print("=" * 72)
    print("  Rosa-sinensis Leaf Health & Severity Classification System")
    print(f"  Mode: {mode_str}")
    print("=" * 72)

    # ── Step 1: Preprocessing ──────────────────────────────────────────
    print("\n[1/6] Loading and preprocessing image(s)...")
    raw_front = load_image(f_path)
    raw_back = load_image(b_path)

    resized_front = resize_to_working_resolution(raw_front)
    resized_back = resize_to_working_resolution(raw_back)

    denoised_front = denoise(resized_front)
    denoised_back = denoise(resized_back)

    prep_front = normalize_brightness(denoised_front)
    prep_back = normalize_brightness(denoised_back)
    print("      Preprocessing complete (denoised & normalized).")

    # ── Step 2: Species Check (Hard Gate) ──────────────────────────────
    print("\n[2/6] Segmenting leaf and verifying Hibiscus species...")
    seg_res = segment_leaf(prep_front)
    initial_mask = seg_res['mask']

    species_res = verify_hibiscus_species(initial_mask)
    if not species_res['is_hibiscus']:
        print(f"      [X] Species Check REJECTED: {species_res['reason']}")
        return {
            'is_hibiscus': False,
            'reason': species_res['reason'],
            'image_path': f_path,
        }
    print("      [OK] Species Verified: Hibiscus rosa-sinensis.")

    # ── Step 3: Interactive ROI Lasso Tracing ──────────────────────────
    mask = initial_mask.copy()
    if use_interactive_roi:
        print("\n[3/6] Interactive Region of Interest (ROI) Selection:")
        print("      >> Opening freehand lasso drawing window...")
        print("      >> Click and drag around the region of interest.")
        print("      >> Press ENTER or SPACE to confirm, or ESC to select full leaf.")
        roi_mask = select_circle_roi(resized_front)
        mask = cv2.bitwise_and(initial_mask, roi_mask)
        print("      ROI mask successfully applied.")
    else:
        print("\n[3/6] Skipping interactive ROI (using full segmented leaf lamina).")

    leaf_area = cv2.countNonZero(mask)
    print(f"      Effective Analyzed Area: {leaf_area:,} pixels")

    # ── Step 4: Vein Extraction ────────────────────────────────────────
    print("\n[4/6] Extracting vein architecture from transmitted light...")
    vein_res = extract_veins(prep_back, mask)
    print(f"      Vein Pixels Detected: {vein_res['vein_pixel_count']:,}")
    print(f"      Branch Junctions:     {vein_res['branch_point_count']}")

    # ── Step 5: Vein Overlay Visual Confirmation ───────────────────────
    if display_overlay:
        print("\n[5/6] Visual Confirmation Window:")
        print("      >> Displaying vein skeleton overlay popup.")
        print("      >> Press ANY KEY in the image window to proceed to severity analysis.")
        window_title = "Vein Skeleton Confirmation (Press any key to proceed)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, vein_res['debug_overlay'])
        cv2.waitKey(0)
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_title)
        except cv2.error:
            pass
    else:
        print("\n[5/6] Vein visual confirmation popup bypassed (--no-display or headless).")

    # ── Step 6: Feature Extraction ─────────────────────────────────────
    print("\n[6/6] Computing botanical chlorosis & vein metric features...")
    features = extract_all_features(prep_front, mask, vein_res, leaf_area)
    print(f"      Vein Density:          {features['vein_density']:.4f}")
    print(f"      Yellow Pixel Ratio:    {features['yellow_pixel_ratio']:.1%}")
    print(f"      DGCI (Dark Green):     {features['dgci']:.3f}")
    print(f"      Interveinal Contrast:  {features['interveinal_contrast']:.1f}")

    # ── Diagnostic & Severity Evaluation ───────────────────────────────
    original_verdict = evaluate(features)
    severity_eval = compute_severity(features)

    # Optional ML & DL predictions
    ml_result = None
    if include_ml:
        try:
            from severityClass.ml_model import predict_leaf
            ml_result = predict_leaf(f_path, model_type='rf')
        except Exception as e:
            ml_result = {'error': str(e)}

    dl_result = None
    if include_dl:
        try:
            from severityClass.dl_model import predict_leaf_dl
            dl_result = predict_leaf_dl(f_path, generate_cam=bool(output_dir))
        except Exception as e:
            dl_result = {'error': str(e)}

    # ── Generate Annotated Visualization ───────────────────────────────
    annotated_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_stem = os.path.splitext(os.path.basename(f_path))[0]
        annotated_path = os.path.join(output_dir, f"{img_stem}_severity_full.jpg")

        overlay = vein_res['debug_overlay'].copy()
        banner_colors = {
            "Healthy": (50, 180, 50),
            "Mild": (50, 200, 240),
            "Moderate": (20, 130, 240),
            "Severe": (40, 40, 220),
        }
        color = banner_colors.get(severity_eval['severity_level'], (128, 128, 128))
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, 80), (w, 85), color, -1)

        title = f"SEVERITY: {severity_eval['severity_level'].upper()} (Score: {severity_eval['severity_score']:.1f}/100)"
        cv2.putText(overlay, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        sub = f"Yellow: {features['yellow_pixel_ratio']:.1%} | DGCI: {features['dgci']:.2f} | Contrast: {features['interveinal_contrast']:.1f} | Density: {features['vein_density']:.4f}"
        cv2.putText(overlay, sub, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imwrite(annotated_path, overlay)

    return {
        'is_hibiscus': True,
        'image_path': f_path,
        'original_verdict': original_verdict,
        'severity': severity_eval,
        'features': features,
        'ml_result': ml_result,
        'dl_result': dl_result,
        'annotated_path': annotated_path,
    }


def print_results(res: Dict[str, Any]):
    """Pretty-print complete diagnosis and severity findings."""
    if not res.get('is_hibiscus', False):
        print("\n" + "=" * 72)
        print(f"  [X] ANALYSIS TERMINATED: Not Hibiscus rosa-sinensis")
        print(f"  Reason: {res.get('reason')}")
        print("=" * 72)
        return

    sev = res['severity']
    orig = res['original_verdict']
    rec = sev['recommendation']

    print("\n" + "=" * 72)
    print("                   DIAGNOSIS & SEVERITY REPORT")
    print("=" * 72)
    print(f"Image File:          {res['image_path']}")
    print(f"Original Status:     {orig['verdict']}")
    print(f"Confidence Signals:  {orig['confidence_signals']} flagged")
    print("-" * 72)
    print(f"SEVERITY GRADE:      {sev['severity_level'].upper()}")
    print(f"SEVERITY SCORE:      {sev['severity_score']:.1f} / 100")
    print(f"Active Thresholds:   {sev['thresholds_used'][sev['severity_level'].lower()]}")
    print("-" * 72)
    print("Component Score Breakdown (0 - 100 scale):")
    for name, val in sev['components'].items():
        print(f"  * {name.replace('_', ' ').title():32s}: {val:5.1f}")
    print("-" * 72)
    print(f"Diagnostic Explanation:\n  {sev['explanation']}")
    print("-" * 72)

    # Multi-model comparison if available
    ml = res.get('ml_result')
    dl = res.get('dl_result')
    if ml or dl:
        print("Model Consensus:")
        print(f"  - Rule-Based Engine:   {sev['severity_level'].upper()} (Score: {sev['severity_score']:.1f})")
        if ml and 'predicted_class' in ml:
            rf_prob = ml['probabilities'].get(ml['predicted_class'], 0.0)
            print(f"  - Random Forest ML:    {ml['predicted_class'].upper()} (Confidence: {rf_prob:.1%})")
        if dl and 'predicted_class' in dl:
            dl_prob = dl['confidence']
            print(f"  - MobileNetV2 DL:      {dl['predicted_class'].upper()} (Confidence: {dl_prob:.1%})")
        print("-" * 72)

    print(f"ACTIONABLE BOTANICAL RECOMMENDATION ({rec['status'].upper()}):")
    print(f"  Primary Focus: {rec['nutrient_focus']}")
    print(f"  Urgency:       {rec['urgency']}")
    print("  Recommended Actions:")
    for step in rec['actions']:
        print(f"    * {step}")

    if res.get('annotated_path'):
        print("-" * 72)
        print(f"Saved Annotated Image: {res['annotated_path']}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rosa-sinensis Leaf Health & Severity Classifier (Full Interactive Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_argument_group('Input Images')
    input_group.add_argument('--image', type=str, default=None, help="Path to single leaf image")
    input_group.add_argument('--backlit', type=str, default=None, help="Path to backlit (transmitted) image")
    input_group.add_argument('--frontlit', type=str, default=None, help="Path to frontlit (incident) image")

    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        '--select-roi', dest='select_roi', action='store_true', default=True,
        help="Interactively select ROI with freehand lasso polygon (default: True)"
    )
    roi_group.add_argument(
        '--no-roi', dest='select_roi', action='store_false',
        help="Skip interactive ROI selection and analyze entire segmented leaf"
    )

    parser.add_argument('--no-display', action='store_true', help="Suppress GUI popup windows (vein confirmation & ROI)")
    parser.add_argument('--output', '-o', type=str, default=None, help="Output directory to save reports and annotated image")
    parser.add_argument('--no-ml', action='store_true', help="Skip ML Random Forest prediction")
    parser.add_argument('--no-dl', action='store_true', help="Skip DL MobileNetV2 prediction")

    args = parser.parse_args()

    # Validate inputs
    has_paired = args.backlit is not None or args.frontlit is not None
    has_single = args.image is not None

    if has_paired and has_single:
        parser.error("Specify either --image OR both --backlit and --frontlit, not both.")
    if has_paired and (args.backlit is None or args.frontlit is None):
        parser.error("Both --backlit and --frontlit are required for paired mode.")
    if not has_paired and not has_single:
        parser.error("Please provide --image <path> or --backlit <path> --frontlit <path>.")

    is_headless = os.environ.get('HEADLESS', '').lower() in ('1', 'true', 'yes')
    use_roi = args.select_roi and not args.no_display and not is_headless
    display_win = not args.no_display and not is_headless

    result = run_severity_pipeline(
        image_path=args.image,
        backlit_path=args.backlit,
        frontlit_path=args.frontlit,
        output_dir=args.output,
        use_interactive_roi=use_roi,
        display_overlay=display_win,
        include_ml=not args.no_ml,
        include_dl=not args.no_dl,
    )

    print_results(result)


if __name__ == '__main__':
    main()
