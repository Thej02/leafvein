"""
scripts/export_preprocessed_steps.py

Processes a sample leaf image step-by-step through all preprocessing, 
segmentation, and vein extraction stages, saving the intermediate images
into the 'preprocessed_images' directory.
"""

import os
import sys
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

from src.preprocessing import (
    load_image,
    resize_to_working_resolution,
    denoise,
    normalize_brightness,
    get_image_stem,
    print_preprocessed_stages_summary,
)
from src.segmentation import segment_leaf
from src.vein_extraction import (
    enhance_vein_contrast,
    extract_veins_adaptive,
    extract_veins_frangi,
    combine_vein_detections,
    skeletonize_veins,
    create_debug_overlay,
)

def main():
    parser = argparse.ArgumentParser(
        description="Processes a leaf image step-by-step through all 12 preprocessing stages, saving them to preprocessed_images/<image_name>/."
    )
    parser.add_argument("image", nargs="?", default=None, help="Path to input leaf image")
    parser.add_argument("--image", "-i", dest="image_opt", default=None, help="Path to input leaf image")
    args = parser.parse_args()

    input_path = args.image or args.image_opt
    if not input_path:
        for candidate in [os.path.join("data", "raw", "1.jpeg"), os.path.join("sample_images", "1.jpeg")]:
            if os.path.exists(candidate):
                input_path = candidate
                break

    if not input_path or not os.path.exists(input_path):
        print(f"Error: Could not find sample image at {input_path}")
        sys.exit(1)

    image_name = get_image_stem(input_path)
    output_dir = os.path.join("preprocessed_images", image_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading raw image from: {input_path}")
    raw_img = load_image(input_path)
    cv2.imwrite(os.path.join(output_dir, "01_raw.jpg"), raw_img)
    
    # 1. Resize
    resized_img = resize_to_working_resolution(raw_img)
    cv2.imwrite(os.path.join(output_dir, "02_resized.jpg"), resized_img)
    
    # 2. Denoise
    denoised_img = denoise(resized_img)
    cv2.imwrite(os.path.join(output_dir, "03_denoised.jpg"), denoised_img)
    
    # 3. Brightness Normalization (Global CLAHE on L channel in LAB)
    normalized_img = normalize_brightness(denoised_img)
    cv2.imwrite(os.path.join(output_dir, "04_normalized.jpg"), normalized_img)
    
    # 4. Leaf Segmentation Mask
    seg_result = segment_leaf(normalized_img)
    mask = seg_result['mask']
    cv2.imwrite(os.path.join(output_dir, "05_leaf_mask.jpg"), mask)
    
    # 5. Grayscale & Masked to Leaf Region
    gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    cv2.imwrite(os.path.join(output_dir, "06_gray_masked.jpg"), gray_masked)
    
    # 6. Local CLAHE Contrast Enhancement
    gray_enhanced = enhance_vein_contrast(gray_masked)
    cv2.imwrite(os.path.join(output_dir, "07_local_enhanced.jpg"), gray_enhanced)
    
    # 7. Adaptive Thresholding
    adaptive_veins = extract_veins_adaptive(gray_enhanced, mask)
    cv2.imwrite(os.path.join(output_dir, "08_adaptive_veins.jpg"), adaptive_veins)
    
    # 8. Frangi Filter Detection
    frangi_veins = extract_veins_frangi(gray_enhanced, mask)
    cv2.imwrite(os.path.join(output_dir, "09_frangi_veins.jpg"), frangi_veins)
    
    # 9. Combined & Morphed
    combined_veins = combine_vein_detections(adaptive_veins, frangi_veins)
    cv2.imwrite(os.path.join(output_dir, "10_combined_veins.jpg"), combined_veins)
    
    # 10. Skeletonization
    skeleton = skeletonize_veins(combined_veins)
    cv2.imwrite(os.path.join(output_dir, "11_skeleton.jpg"), skeleton)
    
    # 11. Final Debug Overlay
    overlay = create_debug_overlay(normalized_img, skeleton, mask)
    cv2.imwrite(os.path.join(output_dir, "12_vein_overlay.jpg"), overlay)
    
    print_preprocessed_stages_summary(output_dir)
    print(f"Done! Successfully saved all 12 transformation stages to: '{output_dir}'")

if __name__ == "__main__":
    main()
