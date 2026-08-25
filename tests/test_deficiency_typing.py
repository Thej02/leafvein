import unittest
from src.deficiency_typing import identify_deficiency
from src.decision_engine import evaluate
from config.thresholds import (
    COLOR_SPATIAL_VARIANCE_MAX,
    INTERVEINAL_CONTRAST_THRESHOLD,
    YELLOW_RATIO_DEFICIENT,
)

class TestDeficiencyTyping(unittest.TestCase):

    def test_pest_confound_pattern(self):
        features = {
            'color_spatial_variance': COLOR_SPATIAL_VARIANCE_MAX + 10,
            'interveinal_contrast': INTERVEINAL_CONTRAST_THRESHOLD + 10,
            'yellow_pixel_ratio': YELLOW_RATIO_DEFICIENT + 0.1
        }
        res = identify_deficiency(features, [])
        self.assertEqual(res['pattern_name'], "Possible pest/disease/damage")
        self.assertIn("Pattern consistent with possible pest/disease/damage", res['assessment_text'])
        self.assertIn("not a substitute", res['assessment_text'])

    def test_iron_magnesium_pattern(self):
        features = {
            'color_spatial_variance': COLOR_SPATIAL_VARIANCE_MAX - 10,
            'interveinal_contrast': INTERVEINAL_CONTRAST_THRESHOLD + 10,
            'yellow_pixel_ratio': YELLOW_RATIO_DEFICIENT + 0.1
        }
        res = identify_deficiency(features, [])
        self.assertEqual(res['pattern_name'], "Iron or magnesium deficiency pattern")
        self.assertIn("Pattern consistent with iron or magnesium deficiency", res['assessment_text'])
        self.assertIn("not a substitute", res['assessment_text'])

    def test_nitrogen_pattern(self):
        features = {
            'color_spatial_variance': COLOR_SPATIAL_VARIANCE_MAX - 10,
            'interveinal_contrast': INTERVEINAL_CONTRAST_THRESHOLD - 10,
            'yellow_pixel_ratio': YELLOW_RATIO_DEFICIENT + 0.1
        }
        res = identify_deficiency(features, [])
        self.assertEqual(res['pattern_name'], "Nitrogen deficiency pattern")
        self.assertIn("Pattern consistent with nitrogen deficiency", res['assessment_text'])
        self.assertIn("not a substitute", res['assessment_text'])

    def test_unknown_pattern(self):
        # High contrast, but NOT high yellow ratio, low variance
        features = {
            'color_spatial_variance': COLOR_SPATIAL_VARIANCE_MAX - 10,
            'interveinal_contrast': INTERVEINAL_CONTRAST_THRESHOLD + 10,
            'yellow_pixel_ratio': YELLOW_RATIO_DEFICIENT - 0.1
        }
        res = identify_deficiency(features, [])
        self.assertEqual(res['pattern_name'], "Unknown pattern")
        self.assertIn("does not clearly match a known deficiency signature", res['assessment_text'])
        self.assertIn("not a substitute", res['assessment_text'])

    def test_not_called_when_healthy(self):
        # A perfectly healthy leaf
        features = {
            'color_spatial_variance': COLOR_SPATIAL_VARIANCE_MAX - 10,
            'interveinal_contrast': INTERVEINAL_CONTRAST_THRESHOLD - 10,
            'yellow_pixel_ratio': YELLOW_RATIO_DEFICIENT - 0.1,
            'excess_green_index': 1.0,
            'dgci': 1.0,
            'vein_density': 1.0,
            'vein_thickness_avg': 1.0
        }
        # In decision engine evaluate, deficiency_assessment should be None if HEALTHY
        result = evaluate(features)
        self.assertEqual(result['verdict'], "Leaf Health Status: HEALTHY")
        self.assertIsNone(result.get('deficiency_assessment'))

if __name__ == '__main__':
    unittest.main()
