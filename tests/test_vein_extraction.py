"""
tests/test_vein_extraction.py — Unit tests for vein extraction.

Tests:
  1. Vein enhancement produces visible output
  2. Skeletonization reduces a thick vein to 1-pixel width
  3. Branch point counting on known patterns
  4. Full pipeline produces expected output structure
"""

import cv2
import numpy as np
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vein_extraction import (
    enhance_vein_contrast,
    extract_veins_adaptive,
    skeletonize_veins,
    count_branch_points,
    extract_veins,
    create_debug_overlay,
)


class TestEnhanceVeinContrast:
    """Tests for CLAHE vein contrast enhancement."""

    def test_output_shape_preserved(self):
        """Output should have same shape as input."""
        gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        enhanced = enhance_vein_contrast(gray)
        assert enhanced.shape == gray.shape

    def test_output_is_uint8(self):
        """Output should be uint8."""
        gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        enhanced = enhance_vein_contrast(gray)
        assert enhanced.dtype == np.uint8


class TestSkeletonize:
    """Tests for vein skeletonization."""

    def test_thick_line_becomes_thin(self):
        """A thick line should be reduced to ~1 pixel width."""
        # Create a 100x100 image with a thick horizontal line (10px wide)
        vein_mask = np.zeros((100, 100), dtype=np.uint8)
        vein_mask[45:55, 10:90] = 255  # 10px wide, 80px long

        skeleton = skeletonize_veins(vein_mask)

        # Skeleton should have pixels (line was long enough to survive pruning)
        assert cv2.countNonZero(skeleton) > 0

        # Skeleton should be much thinner than original
        orig_pixels = cv2.countNonZero(vein_mask)
        skel_pixels = cv2.countNonZero(skeleton)
        assert skel_pixels < orig_pixels * 0.3, (
            f"Skeleton ({skel_pixels}px) should be much thinner than "
            f"original ({orig_pixels}px)"
        )

    def test_empty_input(self):
        """An empty mask should produce an empty skeleton."""
        vein_mask = np.zeros((100, 100), dtype=np.uint8)
        skeleton = skeletonize_veins(vein_mask)
        assert cv2.countNonZero(skeleton) == 0


class TestCountBranchPoints:
    """Tests for branch point counting."""

    def test_straight_line_no_branches(self):
        """A straight line should have 0 branch points."""
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        skeleton[25, 5:45] = 255  # horizontal line

        branches = count_branch_points(skeleton)
        assert branches == 0

    def test_t_junction(self):
        """A T-junction should have at least 1 branch point."""
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        skeleton[25, 5:45] = 255  # horizontal line
        skeleton[25:45, 25] = 255  # vertical branch

        branches = count_branch_points(skeleton)
        assert branches >= 1

    def test_empty_skeleton(self):
        """An empty skeleton should have 0 branch points."""
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        assert count_branch_points(skeleton) == 0


class TestExtractVeins:
    """Integration tests for the full vein extraction pipeline."""

    def test_output_structure(self):
        """Pipeline should return all expected keys."""
        # Create a simple synthetic image with some "veins"
        image = np.ones((200, 200, 3), dtype=np.uint8) * 180  # bright background
        # Draw some dark lines (simulating veins in backlit view)
        cv2.line(image, (20, 100), (180, 100), (40, 40, 40), 3)
        cv2.line(image, (100, 20), (100, 180), (40, 40, 40), 3)
        cv2.line(image, (50, 50), (150, 150), (50, 50, 50), 2)

        # Full mask
        mask = np.ones((200, 200), dtype=np.uint8) * 255

        result = extract_veins(image, mask)

        assert 'vein_mask' in result
        assert 'skeleton' in result
        assert 'vein_pixel_count' in result
        assert 'branch_point_count' in result
        assert 'debug_overlay' in result
        assert isinstance(result['vein_pixel_count'], int)
        assert isinstance(result['branch_point_count'], int)

    def test_empty_mask_returns_zeros(self):
        """With an empty mask, vein extraction should produce zero counts."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = extract_veins(image, mask)

        assert result['vein_pixel_count'] == 0
        assert result['branch_point_count'] == 0


class TestDebugOverlay:
    """Tests for the debug overlay generator."""

    def test_overlay_shape(self):
        """Overlay should have the same dimensions as input image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        skeleton[50, 20:80] = 255
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        overlay = create_debug_overlay(image, skeleton, mask)
        assert overlay.shape == image.shape
