"""
tests/test_preprocessing_stages.py — Tests ensuring all 12 preprocessing stages
are correctly saved into preprocessed_images/<image_name>/ upon running.
"""

import os
import sys
import shutil
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import get_image_stem, print_preprocessed_stages_summary
from src.pipeline import run_pipeline_single_image


def test_get_image_stem():
    assert get_image_stem("sample_images/1.jpeg") == "1"
    assert get_image_stem("data/raw/14.jpeg") == "14"
    assert get_image_stem("leaf_test_backlit.jpg") == "leaf_test"
    assert get_image_stem("path/to/leaf_test_frontlit.png") == "leaf_test"
    assert get_image_stem("hibiscus_sample.jpeg") == "hibiscus_sample"


def test_12_stages_saved_in_pipeline(tmp_path):
    """
    Verify that running an input image saves all 12 stages into preprocessed_images/<id>/
    """
    # Use sample image if available
    sample_path = os.path.join("sample_images", "1.jpeg")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("data", "raw", "1.jpeg")

    assert os.path.exists(sample_path), f"Sample image not found at {sample_path}"

    test_id = "test_verify_12_stages"
    test_dir = os.path.join("preprocessed_images", test_id)

    try:
        # Clean up if exists
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        result = run_pipeline_single_image(
            image_path=sample_path,
            image_id=test_id,
            save_debug=True,
            display_overlay=False,
        )

        assert os.path.isdir(test_dir), f"Directory {test_dir} was not created!"

        expected_stages = [
            "01_raw.jpg",
            "02_resized.jpg",
            "03_denoised.jpg",
            "04_normalized.jpg",
            "05_leaf_mask.jpg",
            "06_gray_masked.jpg",
            "07_local_enhanced.jpg",
            "08_adaptive_veins.jpg",
            "09_frangi_veins.jpg",
            "10_combined_veins.jpg",
            "11_skeleton.jpg",
            "12_vein_overlay.jpg",
        ]

        saved_files = os.listdir(test_dir)
        for stage in expected_stages:
            assert stage in saved_files, f"Stage {stage} is missing from {test_dir}! Found: {saved_files}"
            stage_path = os.path.join(test_dir, stage)
            assert os.path.getsize(stage_path) > 0, f"Stage file {stage} is empty!"

        assert result['preprocessed_dir'] == test_dir
        assert len(result['preprocessed_files']) == 12

    finally:
        # Cleanup test folder
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
