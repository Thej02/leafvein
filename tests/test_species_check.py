import unittest
from unittest.mock import patch
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.species_check import verify_hibiscus_species
from src.pipeline import run_pipeline_single_image

class TestSpeciesCheck(unittest.TestCase):

    def test_verify_hibiscus_species_valid(self):
        # Create a mock valid mask (an ellipse representing an ovate leaf)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 100), (40, 80), 0, 0, 360, 255, -1)
        
        result = verify_hibiscus_species(mask)
        self.assertTrue(result['is_hibiscus'])

    def test_verify_hibiscus_species_invalid_aspect_ratio(self):
        # Create a very long, skinny leaf (like grass)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 100), (5, 90), 0, 0, 360, 255, -1)
        
        result = verify_hibiscus_species(mask)
        self.assertFalse(result['is_hibiscus'])
        self.assertIn("Aspect ratio", result['reason'])

    @patch('src.pipeline.verify_hibiscus_species')
    @patch('src.pipeline.extract_veins')
    @patch('src.pipeline.extract_all_features')
    @patch('src.pipeline.evaluate')
    @patch('src.pipeline.load_image')
    def test_pipeline_aborts_on_invalid_species(self, mock_load, mock_eval, mock_extract_feat, mock_extract_veins, mock_verify):
        # Setup mocks
        mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_verify.return_value = {
            'is_hibiscus': False,
            'reason': 'Aspect ratio (5.00) out of range for Hibiscus rosa-sinensis.'
        }
        
        # Run pipeline
        result = run_pipeline_single_image('dummy_path.jpg', 'dummy')
        
        # Assertions
        # 1. Check that verify was called
        mock_verify.assert_called_once()
        
        # 2. Check that downstream functions were NEVER called (Hard gate)
        mock_extract_veins.assert_not_called()
        mock_extract_feat.assert_not_called()
        mock_eval.assert_not_called()
        
        # 3. Check exact abort result behavior
        self.assertEqual(result['verdict_result']['verdict'], 'Aborted (Not Hibiscus)')
        self.assertIn('Aspect ratio (5.00) out of range', result['verdict_result']['reasoning'])
        self.assertIn('Species Check Failed', result['report_text'])
        self.assertEqual(result['output_files'], [])

if __name__ == '__main__':
    unittest.main()
