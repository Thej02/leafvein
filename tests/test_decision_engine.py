"""
tests/test_decision_engine.py — Unit tests for the rule-based decision engine.

Tests every branch of the decision logic using hand-constructed feature dicts.
This is critical — the decision engine is the core "brain" of the system and
every path must be verified.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_engine import (
    evaluate,
    VERDICT_HEALTHY,
    VERDICT_NOT_HEALTHY,
)
from config.thresholds import (
    YELLOW_RATIO_DEFICIENT,
    EXG_HEALTHY_LOW,
    DGCI_HEALTHY_LOW,
    INTERVEINAL_CONTRAST_THRESHOLD,
    COLOR_SPATIAL_VARIANCE_MAX,
    VEIN_DENSITY_DEFICIENT,
    VEIN_THICKNESS_DEFICIENT_HIGH,
)

def _healthy_features():
    """Return a feature dict representing a clearly healthy leaf."""
    return {
        'vein_density': 0.05,               # above threshold
        'vein_thickness_avg': 7.0,          # below high threshold
        'yellow_pixel_ratio': 0.05,         # minimal yellow
        'excess_green_index': 0.65,         # good green
        'dgci': 0.55,                       # good DGCI
        'interveinal_contrast': 30.0,       # below threshold
        'color_spatial_variance': 400.0,    # below threshold
    }

def _deficient_features():
    """Return a feature dict representing a clearly deficient leaf."""
    return {
        'vein_density': 0.01,                # below deficient threshold (secondary)
        'vein_thickness_avg': 10.0,          # above healthy (secondary)
        'yellow_pixel_ratio': 0.20,          # lots of yellow (primary)
        'excess_green_index': 0.40,          # low green (primary)
        'dgci': 0.35,                        # low DGCI (primary)
        'interveinal_contrast': 60.0,        # high contrast (primary)
        'color_spatial_variance': 800.0,     # high variance (primary)
    }

class TestHealthyVerdict(unittest.TestCase):
    def test_clearly_healthy(self):
        """All features within healthy range → Healthy verdict."""
        result = evaluate(_healthy_features())
        self.assertEqual(result['verdict'], VERDICT_HEALTHY)
        self.assertEqual(result['confidence_signals'], 0)
        self.assertEqual(len(result['flags']), 0)

class TestNotHealthyVerdict(unittest.TestCase):
    def test_clearly_deficient(self):
        """All features abnormal → NOT HEALTHY verdict."""
        result = evaluate(_deficient_features())
        self.assertEqual(result['verdict'], VERDICT_NOT_HEALTHY)
        self.assertGreater(result['confidence_signals'], 0)

    def test_single_primary_flag(self):
        """One primary feature out of range → NOT HEALTHY."""
        features = _healthy_features()
        features['yellow_pixel_ratio'] = YELLOW_RATIO_DEFICIENT + 0.05

        result = evaluate(features)
        self.assertEqual(result['verdict'], VERDICT_NOT_HEALTHY)
        self.assertEqual(len(result['flags']), 1)
        self.assertEqual(result['flags'][0]['type'], 'primary')

class TestSecondaryFactorRegression(unittest.TestCase):
    def test_secondary_only_failure(self):
        """
        Regression test: If primary factors are healthy, 
        a secondary factor failure MUST NOT trigger an unhealthy verdict.
        """
        features = _healthy_features()
        # Fail the secondary factors
        features['vein_density'] = VEIN_DENSITY_DEFICIENT - 0.01
        features['vein_thickness_avg'] = VEIN_THICKNESS_DEFICIENT_HIGH + 1.0

        result = evaluate(features)
        
        # Primary factors are all healthy, so verdict should be healthy
        self.assertEqual(result['verdict'], VERDICT_HEALTHY)
        
        # And secondary flags should NOT be returned/included
        self.assertEqual(len(result['flags']), 0)
        self.assertEqual(result['confidence_signals'], 0)

class TestFlagStructure(unittest.TestCase):
    def test_flag_keys(self):
        """Each flag should have the required keys."""
        result = evaluate(_deficient_features())
        self.assertGreater(len(result['flags']), 0)

        for flag in result['flags']:
            self.assertIn('feature', flag)
            self.assertIn('value', flag)
            self.assertIn('threshold', flag)
            self.assertIn('type', flag)
            self.assertIn('meaning', flag)
            self.assertIn(flag['type'], ('primary', 'secondary'))

if __name__ == '__main__':
    unittest.main()
