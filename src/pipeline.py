"""
src/pipeline.py — End-to-end orchestration of the leaf health detection pipeline.

Chains together:
  1. Preprocessing (both images)
  2. Segmentation (both images — same leaf, same mask approach)
  3. Vein extraction (backlit image)
  4. Feature extraction (both images)
  5. Decision engine
  6. Report generation

This module is called by cli.py and provides the complete analysis workflow.
"""

import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import (
    load_image,
    resize_to_working_resolution,
    denoise,
    normalize_brightness,
    preprocess
)
from src.segmentation import segment_leaf
from src.species_check import verify_hibiscus_species
from src.vein_extraction import extract_veins
from src.feature_extraction import extract_all_features
from src.decision_engine import evaluate
from src.report_generator import (
    generate_text_report,
    generate_annotated_image,
    save_report,
    save_annotated_image,
    generate_unhealthy_regions_image,
)
from src.interactive_roi import select_circle_roi


def run_pipeline(backlit_path: str,
                  frontlit_path: str,
                  image_id: str = None,
                  output_dir: str = None,
                  save_debug: bool = True,
                  use_interactive_roi: bool = False) -> dict:
    """
    Run the complete leaf health analysis pipeline.

    Args:
        backlit_path: Path to the backlit (transmitted light) leaf image.
        frontlit_path: Path to the front-lit (daylight) leaf image.
        image_id: Optional identifier for the leaf. Defaults to filename stem.
        output_dir: Directory to save reports and debug images. If None, only
                    returns results without saving.
        save_debug: If True, save debug overlays and annotated images.

    Returns:
        Dict with keys:
          'image_id': str
          'features': dict — all extracted feature values
          'verdict_result': dict — verdict, reasoning, flags
          'segmentation_validation': dict — mask quality info
          'report_text': str — formatted text report
          'output_files': list of str — paths to saved files (if output_dir set)
    """
    output_files = []

    # Default image ID from filename
    if image_id is None:
        image_id = os.path.splitext(os.path.basename(backlit_path))[0]
        # Strip common suffixes
        for suffix in ['_backlit', '_frontlit', '_back', '_front']:
            if image_id.endswith(suffix):
                image_id = image_id[:-len(suffix)]
                break

    # ── Step 1: Load and preprocess ────────────────────────────────────
    print(f"[1/6] Loading and preprocessing images...")

    backlit_raw = load_image(backlit_path)
    frontlit_raw = load_image(frontlit_path)

    backlit_resized = resize_to_working_resolution(backlit_raw)
    frontlit_resized = resize_to_working_resolution(frontlit_raw)

    backlit_denoised = denoise(backlit_resized)
    frontlit_denoised = denoise(frontlit_resized)

    backlit_preprocessed = normalize_brightness(backlit_denoised)
    frontlit_preprocessed = normalize_brightness(frontlit_denoised)

    print(f"       Backlit:  {backlit_raw.shape} -> {backlit_preprocessed.shape}")
    print(f"       Frontlit: {frontlit_raw.shape} -> {frontlit_preprocessed.shape}")

    # Create folder for this execution inside preprocessed_images
    prep_dir = os.path.join("preprocessed_images", image_id)
    os.makedirs(prep_dir, exist_ok=True)
    cv2.imwrite(os.path.join(prep_dir, "01_raw.jpg"), backlit_raw)
    cv2.imwrite(os.path.join(prep_dir, "02_resized.jpg"), backlit_resized)
    cv2.imwrite(os.path.join(prep_dir, "03_denoised.jpg"), backlit_denoised)
    cv2.imwrite(os.path.join(prep_dir, "04_normalized.jpg"), backlit_preprocessed)

    # ── Step 2: Segment leaf ──────────────────────────────────────────
    print(f"[2/6] Segmenting leaf from background...")

    seg_result = segment_leaf(frontlit_preprocessed)
    mask = seg_result['mask']
    validation = seg_result['validation']

    if not validation['valid']:
        for issue in validation['issues']:
            print(f"       ! {issue}")

    # ── Step 1.5: Species Check (Hard Gate) ───────────────────────────
    species_res = verify_hibiscus_species(mask)
    if not species_res['is_hibiscus']:
        print("Species Check: The leaf is NOT Hibiscus (Rosa-sinensis). Analysis aborted.")
        print(f"       Reason: {species_res['reason']}")
        return {
            'image_id': image_id,
            'verdict_result': {
                'verdict': 'Aborted (Not Hibiscus)', 
                'reasoning': species_res['reason'], 
                'confidence_signals': 0
            },
            'report_text': f"Species Check Failed:\nThe leaf is NOT Hibiscus (Rosa-sinensis). Analysis aborted.\nReason: {species_res['reason']}",
            'output_files': []
        }
    else:
        print("Species Check: The leaf is Hibiscus (Rosa-sinensis). Proceeding with analysis.")

    # ── Step 2: Interactive ROI Tracing ───────────────────────────────
    if use_interactive_roi:
        print(f"       Opening interactive ROI selection window...")
        roi_mask = select_circle_roi(frontlit_resized)
        mask = cv2.bitwise_and(mask, roi_mask)
        
    cv2.imwrite(os.path.join(prep_dir, "05_leaf_mask.jpg"), mask)

    leaf_area = cv2.countNonZero(mask)
    
    # Update leaf area fraction based on ROI-intersected mask
    h, w = mask.shape
    validation['leaf_area_pixels'] = leaf_area
    validation['leaf_area_fraction'] = leaf_area / (h * w)
    print(f"       Leaf area: {leaf_area:,} pixels ({validation['leaf_area_fraction']:.1%} of frame)")

    # ── Step 3: Extract veins from backlit image ──────────────────────
    print(f"[3/6] Extracting vein architecture from backlit image...")

    vein_result = extract_veins(backlit_preprocessed, mask, save_stages_dir=prep_dir)

    print(f"       Vein pixels: {vein_result['vein_pixel_count']:,}")
    print(f"       Branch points: {vein_result['branch_point_count']}")

    # Save and show vein overlay confirmation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        overlay_path = os.path.join(output_dir, f"{image_id}_vein_overlay.jpg")
        cv2.imwrite(overlay_path, vein_result['debug_overlay'])
        output_files.append(overlay_path)

    print(f"       Displaying vein overlay for visual confirmation...")
    window_name = "Vein Skeleton Confirmation (Press any key to continue)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vein_result['debug_overlay'])
    cv2.waitKey(0)
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    # ── Step 4: Extract features ──────────────────────────────────────
    print(f"[4/6] Computing feature values...")

    features = extract_all_features(frontlit_preprocessed, mask, vein_result, leaf_area)

    print(f"       Vein density: {features['vein_density']:.6f}")
    print(f"       Yellow ratio: {features['yellow_pixel_ratio']:.2%}")
    print(f"       ExG: {features['excess_green_index']:.4f}")
    print(f"       DGCI: {features['dgci']:.4f}")

    # ── Step 5: Run decision engine ───────────────────────────────────
    print(f"[5/6] Evaluating health verdict...")

    verdict_result = evaluate(features)

    print(f"       Verdict: {verdict_result['verdict']}")
    print(f"       Signals: {verdict_result['confidence_signals']} flagged")

    # ── Step 6: Generate report ───────────────────────────────────────
    print(f"[6/6] Generating report...")

    report_text = generate_text_report(image_id, verdict_result, features, validation)

    # Save outputs if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save text report
        report_path = os.path.join(output_dir, f"{image_id}_report.txt")
        save_report(report_text, report_path)
        output_files.append(report_path)
        print(f"       Report saved: {report_path}")

        if save_debug:
            # Save annotated composite image
            annotated = generate_annotated_image(
                frontlit_preprocessed, vein_result['debug_overlay'],
                verdict_result, features, image_id
            )
            annotated_path = os.path.join(output_dir, f"{image_id}_annotated.jpg")
            save_annotated_image(annotated, annotated_path)
            output_files.append(annotated_path)
            print(f"       Annotated image saved: {annotated_path}")

            # Save segmentation mask
            mask_path = os.path.join(output_dir, f"{image_id}_mask.jpg")
            cv2.imwrite(mask_path, mask)
            output_files.append(mask_path)
            
            # Save unhealthy regions image if NOT HEALTHY
            if verdict_result['verdict'] != "Leaf Health Status: HEALTHY":
                unhealthy_regions_img = generate_unhealthy_regions_image(
                    vein_result['debug_overlay'], features, verdict_result
                )
                if unhealthy_regions_img is not None:
                    unhealthy_path = os.path.join(output_dir, f"{image_id}_unhealthy_regions.jpg")
                    cv2.imwrite(unhealthy_path, unhealthy_regions_img)
                    output_files.append(unhealthy_path)
                    print(f"       Unhealthy regions saved: {unhealthy_path}")

    print(f"\nDone. Verdict: {verdict_result['verdict']}")

    return {
        'image_id': image_id,
        'features': features,
        'verdict_result': verdict_result,
        'segmentation_validation': validation,
        'report_text': report_text,
        'output_files': output_files,
    }


def run_pipeline_single_image(image_path: str,
                                image_id: str = None,
                                output_dir: str = None,
                                save_debug: bool = True,
                                use_interactive_roi: bool = False) -> dict:
    """
    Run the pipeline using a single image for both backlit and front-lit analysis.

    This is a convenience function for cases where only one image is available
    (e.g., during initial testing with the sample images). The same image is used
    for both vein extraction and color analysis.

    NOTE: For accurate results, use run_pipeline() with separate backlit and
    front-lit images as specified in the capture protocol.

    Args:
        image_path: Path to the leaf image.
        image_id: Optional identifier. Defaults to filename stem.
        output_dir: Directory for output files.
        save_debug: Whether to save debug images.

    Returns:
        Same result dict as run_pipeline().
    """
    print("NOTE: Single-image mode — using same image for both vein and color analysis.")
    print("      For best results, capture separate backlit and front-lit images.")
    print("")
    return run_pipeline(image_path, image_path, image_id, output_dir, save_debug, use_interactive_roi)
