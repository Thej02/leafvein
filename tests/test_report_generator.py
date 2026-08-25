import cv2
import numpy as np

from src.report_generator import generate_unhealthy_regions_image

def test_unhealthy_regions_circling_healthy_verdict():
    overlay = np.zeros((100, 100, 3), dtype=np.uint8)
    features = {}
    verdict_result = {'verdict': 'Leaf Health Status: HEALTHY'}
    
    result = generate_unhealthy_regions_image(overlay, features, verdict_result)
    assert result is None, "Should return None for HEALTHY verdict"

def test_unhealthy_regions_circling_not_healthy_verdict():
    overlay = np.zeros((100, 100, 3), dtype=np.uint8)
    # create a 20x20 mask area
    yellow_mask = np.zeros((100, 100), dtype=np.uint8)
    yellow_mask[40:60, 40:60] = 255
    
    features = {
        'masks': {
            'yellow_pixel_ratio': yellow_mask
        }
    }
    verdict_result = {
        'verdict': 'Leaf Health Status: NOT HEALTHY',
        'flags': [
            {'feature': 'yellow_pixel_ratio', 'type': 'primary'}
        ]
    }
    
    result = generate_unhealthy_regions_image(overlay, features, verdict_result)
    assert result is not None, "Should return an image for NOT HEALTHY verdict"
    assert result.shape == (100, 100, 3)
    
    # Check that circles were drawn (magenta color 255, 0, 255)
    # In BGR, magenta is [255, 0, 255]
    magenta_pixels = np.all(result == [255, 0, 255], axis=-1)
    assert np.any(magenta_pixels), "Should draw magenta circles on the image"

if __name__ == "__main__":
    test_unhealthy_regions_circling_healthy_verdict()
    test_unhealthy_regions_circling_not_healthy_verdict()
    print("All tests passed!")
