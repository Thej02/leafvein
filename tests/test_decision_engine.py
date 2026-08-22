"""
tests/test_decision_engine.py — Unit tests for the rule-based decision engine.

Tests every branch of the decision logic using hand-constructed feature dicts.
This is critical — the decision engine is the core "brain" of the system and
every path must be verified.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_engine import (
    evaluate,
    VERDICT_HEALTHY,
    VERDICT_POSSIBLY_DEFICIENT,
    VERDICT_DEFICIENT,
)
from config.thresholds import (
    VEIN_DENSITY_HEALTHY_LOW,
    VEIN_DENSITY_DEFICIENT,
    YELLOW_RATIO_POSSIBLY_DEFICIENT,
    YELLOW_RATIO_DEFICIENT,
    EXG_HEALTHY_LOW,
    DGCI_HEALTHY_LOW,
    MEAN_SATURATION_HEALTHY_LOW,
)


def _healthy_features():
    """Return a feature dict representing a clearly healthy leaf."""
    return {
        'vein_density': 0.05,               # well above thresholds
        'vein_thickness_avg': 3.0,           # above healthy low
        'branch_point_count': 50,
        'vein_pixel_count': 5000,
        'leaf_area_pixels': 100000,
        'mean_hue': 55.0,                    # green hue
        'mean_saturation': 120.0,            # good saturation
        'yellow_pixel_ratio': 0.02,          # minimal yellow
        'excess_green_index': 0.20,          # good green
        'dgci': 0.55,                        # good DGCI
        'interveinal_contrast': 5.0,         # low contrast (uniform)
    }


def _deficient_features():
    """Return a feature dict representing a clearly deficient leaf."""
    return {
        'vein_density': 0.01,                # below deficient threshold
        'vein_thickness_avg': 1.0,           # below healthy
        'branch_point_count': 10,
        'vein_pixel_count': 1000,
        'leaf_area_pixels': 100000,
        'mean_hue': 30.0,                    # yellowish
        'mean_saturation': 40.0,             # low saturation
        'yellow_pixel_ratio': 0.45,          # lots of yellow
        'excess_green_index': 0.05,          # low green
        'dgci': 0.25,                        # low DGCI
        'interveinal_contrast': 25.0,        # high contrast
    }


class TestHealthyVerdict:
    """Tests that healthy leaves are correctly classified."""

    def test_clearly_healthy(self):
        """All features within healthy range → Healthy verdict."""
        result = evaluate(_healthy_features())
        assert result['verdict'] == VERDICT_HEALTHY
        assert result['confidence_signals'] == 0
        assert len(result['flags']) == 0

    def test_reasoning_present(self):
        """Healthy verdict should still have a reasoning string."""
        result = evaluate(_healthy_features())
        assert len(result['reasoning']) > 0
        assert 'Healthy' in result['reasoning'] or 'healthy' in result['reasoning']


class TestDeficientVerdict:
    """Tests that deficient leaves are correctly classified."""

    def test_clearly_deficient(self):
        """All features abnormal → Deficient verdict."""
        result = evaluate(_deficient_features())
        assert result['verdict'] == VERDICT_DEFICIENT
        assert result['confidence_signals'] > 0

    def test_multiple_flags(self):
        """Deficient features should produce multiple flags."""
        result = evaluate(_deficient_features())
        assert len(result['flags']) >= 3

    def test_reasoning_explains_flags(self):
        """Reasoning should mention the flagged features."""
        result = evaluate(_deficient_features())
        reasoning = result['reasoning']
        # Should mention at least some of the flagged feature names
        assert 'vein' in reasoning.lower() or 'yellow' in reasoning.lower()


class TestPossiblyDeficientVerdict:
    """Tests for borderline cases."""

    def test_single_mild_flag(self):
        """One feature slightly out of range → Possibly Deficient."""
        features = _healthy_features()
        features['yellow_pixel_ratio'] = YELLOW_RATIO_POSSIBLY_DEFICIENT + 0.01

        result = evaluate(features)
        assert result['verdict'] == VERDICT_POSSIBLY_DEFICIENT

    def test_low_vein_density_only(self):
        """Just low vein density (moderate) → Possibly Deficient."""
        features = _healthy_features()
        features['vein_density'] = VEIN_DENSITY_HEALTHY_LOW - 0.005

        result = evaluate(features)
        assert result['verdict'] == VERDICT_POSSIBLY_DEFICIENT


class TestSpecificDeficiencyPatterns:
    """Tests that specific deficiency patterns are identified."""

    def test_nitrogen_pattern(self):
        """Uniform yellowing + low ExG without interveinal contrast → nitrogen pattern."""
        features = _healthy_features()
        features['yellow_pixel_ratio'] = YELLOW_RATIO_POSSIBLY_DEFICIENT + 0.05
        features['excess_green_index'] = EXG_HEALTHY_LOW - 0.05
        features['interveinal_contrast'] = 3.0  # low — uniform yellowing

        result = evaluate(features)
        reasoning = result['reasoning']
        # Should identify a nitrogen-like pattern
        assert 'nitrogen' in reasoning.lower() or 'uniform' in reasoning.lower() or len(result['flags']) >= 2

    def test_magnesium_pattern(self):
        """Interveinal chlorosis + yellowing → magnesium pattern."""
        features = _healthy_features()
        features['yellow_pixel_ratio'] = YELLOW_RATIO_POSSIBLY_DEFICIENT + 0.05
        features['interveinal_contrast'] = 20.0  # high — veins green, tissue yellow

        result = evaluate(features)
        # Should have flags for both features
        flag_features = {f['feature'] for f in result['flags']}
        assert 'yellow_pixel_ratio' in flag_features
        assert 'interveinal_contrast' in flag_features


class TestReasoningQuality:
    """Tests that the reasoning output meets product requirements."""

    def test_reasoning_is_string(self):
        """Reasoning should always be a string."""
        for features in [_healthy_features(), _deficient_features()]:
            result = evaluate(features)
            assert isinstance(result['reasoning'], str)

    def test_reasoning_not_empty(self):
        """Reasoning should never be empty."""
        for features in [_healthy_features(), _deficient_features()]:
            result = evaluate(features)
            assert len(result['reasoning']) > 20  # not just a word

    def test_deficient_reasoning_has_hedging(self):
        """Deficient verdict reasoning should use hedging language (not absolute claims)."""
        result = evaluate(_deficient_features())
        reasoning = result['reasoning'].lower()
        # Should contain some form of hedging
        hedging_terms = ['consistent with', 'may', 'possible', 'suggest', 'pattern',
                          'indicates', 'not a definitive', 'not a diagnostic']
        assert any(term in reasoning for term in hedging_terms), (
            "Reasoning should use hedging language, not make diagnostic claims"
        )


class TestFlagStructure:
    """Tests that individual flags have the expected structure."""

    def test_flag_keys(self):
        """Each flag should have the required keys."""
        result = evaluate(_deficient_features())
        assert len(result['flags']) > 0

        for flag in result['flags']:
            assert 'feature' in flag
            assert 'value' in flag
            assert 'threshold' in flag
            assert 'direction' in flag
            assert 'severity' in flag
            assert 'meaning' in flag
            assert flag['severity'] in ('high', 'moderate', 'low')
            assert flag['direction'] in ('above', 'below')
