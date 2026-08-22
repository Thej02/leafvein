"""
tests/test_segmentation.py — Unit tests for leaf segmentation.

Tests:
  1. Mask covers a reasonable fraction of a known sample image
  2. Mask produces exactly one connected component
  3. Leaf area count is positive and within expected range
  4. Validation correctly flags edge cases
"""

import cv2
import numpy as np
import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_from_path, preprocess
from src.segmentation import (
    create_leaf_mask,
    apply_mask,
    compute_leaf_area,
    validate_segmentation,
    segment_leaf,
)


# Path to sample images
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sample_images')


def get_sample_images():
    """Get list of available sample image paths."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return [
        os.path.join(SAMPLE_DIR, f)
        for f in sorted(os.listdir(SAMPLE_DIR))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]


class TestCreateLeafMask:
    """Tests for the create_leaf_mask function."""

    def test_synthetic_green_on_black(self):
        """A green rectangle on a black background should be fully segmented."""
        # Create a 200x300 image: green rectangle on black
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 60:240] = (0, 150, 0)  # BGR green

        mask = create_leaf_mask(img)

        # The green region should be captured
        green_region = mask[40:160, 60:240]
        green_coverage = cv2.countNonZero(green_region) / (120 * 180)
        assert green_coverage > 0.7, f"Green region coverage too low: {green_coverage:.1%}"

        # Background should be mostly zero
        bg_region = mask[:40, :]
        bg_coverage = cv2.countNonZero(bg_region) / (40 * 300)
        assert bg_coverage < 0.1, f"Background leakage too high: {bg_coverage:.1%}"

    def test_output_is_binary(self):
        """Mask should only contain 0 and 255 values."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = (0, 128, 0)

        mask = create_leaf_mask(img)

        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, 255}), f"Non-binary values in mask: {unique_values}"


class TestSegmentLeaf:
    """Integration tests for the full segmentation pipeline."""

    @pytest.mark.skipif(len(get_sample_images()) == 0, reason="No sample images available")
    def test_sample_images_segment(self):
        """Each sample image should produce a valid segmentation."""
        sample_images = get_sample_images()[:5]  # Test first 5

        for img_path in sample_images:
            img = preprocess_from_path(img_path)
            result = segment_leaf(img)

            fname = os.path.basename(img_path)

            # Mask should exist and be non-empty
            assert result['mask'] is not None, f"{fname}: mask is None"
            assert result['leaf_area_pixels'] > 0, f"{fname}: no leaf area detected"

            # Leaf should cover a reasonable fraction (at least 5% of frame)
            total_pixels = img.shape[0] * img.shape[1]
            fraction = result['leaf_area_pixels'] / total_pixels
            assert fraction > 0.05, (
                f"{fname}: leaf area fraction ({fraction:.1%}) is suspiciously low"
            )

            # Should have exactly 1 connected component after cleanup
            validation = result['validation']
            assert validation['num_components'] == 1, (
                f"{fname}: expected 1 component, got {validation['num_components']}"
            )

    def test_all_black_image(self):
        """An all-black image should produce an empty mask."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = segment_leaf(img)

        assert result['leaf_area_pixels'] == 0
        assert not result['validation']['valid']


class TestComputeLeafArea:
    """Tests for the compute_leaf_area function."""

    def test_known_area(self):
        """A mask with a known number of white pixels should return that count."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:60] = 255  # 50x50 = 2500 pixels

        area = compute_leaf_area(mask)
        assert area == 2500

    def test_empty_mask(self):
        """An empty mask should have zero area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_leaf_area(mask) == 0


class TestValidateSegmentation:
    """Tests for segmentation validation."""

    def test_valid_segmentation(self):
        """A good mask should pass validation."""
        # 100x100 image, mask covering 30% → above default threshold
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 25:75] = 255  # 60×50 = 3000/10000 = 30%

        result = validate_segmentation(mask, (100, 100, 3))
        assert result['valid'] is True
        assert len(result['issues']) == 0

    def test_too_small_leaf(self):
        """A very small mask should fail validation."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255  # 100 pixels = 1% of 10000

        result = validate_segmentation(mask, (100, 100, 3))
        assert result['valid'] is False
        assert any('below minimum' in issue for issue in result['issues'])
