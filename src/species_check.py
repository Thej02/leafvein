import cv2
import numpy as np

def verify_hibiscus_species(mask: np.ndarray) -> dict:
    """
    Verifies if the segmented leaf matches the morphological characteristics
    of Hibiscus rosa-sinensis (ovate/broadly lanceolate shape).
    
    Args:
        mask: Binary mask of the segmented leaf (255 for leaf, 0 for background).
        
    Returns:
        Dict with keys:
            'is_hibiscus': bool,
            'reason': str
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'is_hibiscus': False, 'reason': 'No leaf contour found.'}
        
    # Get the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    
    if area < 1000:
        return {'is_hibiscus': False, 'reason': 'Leaf area too small for analysis.'}
        
    # Morphological features
    # 1. Aspect Ratio using minimum area rectangle (handles rotation)
    rect = cv2.minAreaRect(main_contour)
    (x, y), (w, h), angle = rect
    
    if min(w, h) == 0:
        return {'is_hibiscus': False, 'reason': 'Invalid leaf dimensions.'}
        
    aspect_ratio = max(w, h) / min(w, h)
    
    # 2. Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 3. Extent (Area / Bounding Box Area)
    extent = float(area) / (w * h) if (w * h) > 0 else 0

    # Hibiscus rosa-sinensis typical bounds (broadly ovate)
    # Aspect Ratio: Typically not perfectly round, nor extremely elongated (like grass)
    # Solidity: Margins are serrated but overall shape is solid (not deeply lobed like maple)
    # Extent: Fills a reasonable portion of its bounding box
    
    if not (1.05 <= aspect_ratio <= 3.0):
        return {
            'is_hibiscus': False, 
            'reason': f'Aspect ratio ({aspect_ratio:.2f}) out of range for Hibiscus rosa-sinensis.'
        }
        
    if solidity < 0.70:
        return {
            'is_hibiscus': False, 
            'reason': f'Leaf solidity ({solidity:.2f}) too low (indicates deep lobes or non-Hibiscus shape).'
        }
        
    if extent < 0.40:
        return {
            'is_hibiscus': False, 
            'reason': f'Leaf extent ({extent:.2f}) too low.'
        }
        
    return {
        'is_hibiscus': True,
        'reason': 'Shape matches Hibiscus rosa-sinensis morphology.'
    }
