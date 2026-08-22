"""
tests/test_feature_extraction.py — Unit tests for feature extraction.

Tests each feature function independently with synthetic images where
expected values are known.
"""

import cv2
import numpy as np
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extraction import (
    compute_vein_density,
    compute_vein_thickness,
    compute_mean_hue_saturation,
    compute_yellow_pixel_ratio,
    compute_excess_green_index,
    compute_dgci,
    compute_interveinal_contrast,
    extract_all_features,
)


class TestVeinDensity:
    """Tests for compute_vein_density."""

    def test_basic_ratio(self):
        """Density should be vein_pixels / leaf_area."""
        assert compute_vein_density(100, 1000) == pytest.approx(0.1)

    def test_zero_leaf_area(self):
        """Zero leaf area should return 0.0, not division error."""
        assert compute_vein_density(100, 0) == 0.0

    def test_zero_veins(self):
        """No vein pixels should return 0.0."""
        assert compute_vein_density(0, 1000) == 0.0


class TestVeinThickness:
    """Tests for compute_vein_thickness."""

    def test_known_thickness(self):
        """A line of known width should produce approximately that thickness."""
        # Create a horizontal band 10 pixels wide
        vein_mask = np.zeros((100, 100), dtype=np.uint8)
        vein_mask[45:55, 10:90] = 255

        # Create a skeleton through the middle
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        skeleton[50, 10:90] = 255

        thickness = compute_vein_thickness(vein_mask, skeleton)

        # Distance from center to edge is ~5, so thickness should be ~10
        assert 4.0 < thickness < 14.0, f"Expected ~10px thickness, got {thickness:.1f}"

    def test_empty_skeleton(self):
        """Empty skeleton should return 0.0."""
        vein_mask = np.zeros((50, 50), dtype=np.uint8)
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        assert compute_vein_thickness(vein_mask, skeleton) == 0.0


class TestMeanHueSaturation:
    """Tests for compute_mean_hue_saturation."""

    def test_pure_green(self):
        """A pure green image should have hue around 60 (120° / 2 in OpenCV)."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 255, 0)  # BGR pure green
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        result = compute_mean_hue_saturation(img, mask)

        # Green hue in OpenCV HSV is around 60 (= 120° / 2)
        assert 50 < result['mean_hue'] < 70, f"Expected hue ~60, got {result['mean_hue']}"
        assert result['mean_saturation'] > 200, f"Expected high saturation, got {result['mean_saturation']}"

    def test_empty_mask(self):
        """Empty mask should return 0.0 for both."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)

        result = compute_mean_hue_saturation(img, mask)
        assert result['mean_hue'] == 0.0
        assert result['mean_saturation'] == 0.0


class TestYellowPixelRatio:
    """Tests for compute_yellow_pixel_ratio."""

    def test_no_yellow(self):
        """A pure green image should have ~0% yellow pixels."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 200, 0)  # BGR green
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        ratio = compute_yellow_pixel_ratio(img, mask)
        assert ratio < 0.1, f"Expected <10% yellow in green image, got {ratio:.1%}"

    def test_yellow_image(self):
        """A yellow image should have a high yellow pixel ratio."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 200, 200)  # BGR yellow-ish
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        ratio = compute_yellow_pixel_ratio(img, mask)
        assert ratio > 0.5, f"Expected >50% yellow in yellow image, got {ratio:.1%}"

    def test_empty_mask(self):
        """Empty mask should return 0.0."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert compute_yellow_pixel_ratio(img, mask) == 0.0


class TestExcessGreenIndex:
    """Tests for compute_excess_green_index."""

    def test_green_positive_exg(self):
        """A green image should have positive ExG."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 200, 50)  # BGR: low B, high G, low R
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        exg = compute_excess_green_index(img, mask)
        assert exg > 0, f"Expected positive ExG for green image, got {exg:.4f}"

    def test_red_negative_exg(self):
        """A red image should have negative ExG."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 0, 200)  # BGR: no B/G, high R
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        exg = compute_excess_green_index(img, mask)
        assert exg < 0, f"Expected negative ExG for red image, got {exg:.4f}"

    def test_empty_mask(self):
        """Empty mask should return 0.0."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert compute_excess_green_index(img, mask) == 0.0


class TestDGCI:
    """Tests for compute_dgci."""

    def test_returns_float(self):
        """DGCI should return a float value."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = (0, 150, 0)
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        dgci = compute_dgci(img, mask)
        assert isinstance(dgci, float)

    def test_empty_mask(self):
        """Empty mask should return 0.0."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert compute_dgci(img, mask) == 0.0


class TestInterveinalContrast:
    """Tests for compute_interveinal_contrast."""

    def test_uniform_color_low_contrast(self):
        """A uniformly colored image should have low interveinal contrast."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 150, 0)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        skeleton[50, 20:80] = 255

        contrast = compute_interveinal_contrast(img, mask, skeleton)
        assert contrast < 5.0, f"Expected low contrast for uniform image, got {contrast}"

    def test_no_skeleton(self):
        """Empty skeleton should return 0.0."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        skeleton = np.zeros((50, 50), dtype=np.uint8)

        assert compute_interveinal_contrast(img, mask, skeleton) == 0.0


class TestExtractAllFeatures:
    """Integration test for extract_all_features."""

    def test_output_keys(self):
        """Should return all expected feature keys."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        vein_result = {
            'skeleton': np.zeros((100, 100), dtype=np.uint8),
            'vein_mask': np.zeros((100, 100), dtype=np.uint8),
            'vein_pixel_count': 0,
            'branch_point_count': 0,
        }

        features = extract_all_features(img, mask, vein_result, 10000)

        expected_keys = {
            'vein_density', 'vein_thickness_avg', 'branch_point_count',
            'vein_pixel_count', 'leaf_area_pixels',
            'mean_hue', 'mean_saturation', 'yellow_pixel_ratio',
            'excess_green_index', 'dgci', 'interveinal_contrast',
        }
        assert set(features.keys()) == expected_keys
