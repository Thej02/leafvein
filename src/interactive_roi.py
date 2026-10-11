import cv2
import numpy as np

def select_circle_roi(image: np.ndarray) -> np.ndarray:
    """
    Opens an interactive window to allow the user to draw a freehand polygon
    (lasso) over the region of interest.
    
    (Note: The function name is kept as select_circle_roi to avoid breaking
     existing pipeline imports, but it now performs freehand lasso selection).
    
    Args:
        image: The image on which to draw the ROI (e.g., preprocessed front-lit).
        
    Returns:
        A binary mask (numpy array of same H, W as image, type uint8) where
        pixels inside the drawn region are 255 and outside are 0.
        If the user cancels (presses ESC) or closes without drawing, returns
        a mask of all 255s (i.e., select everything).
    """
    display_img = image.copy()
    h, w = image.shape[:2]
    
    drawing = False
    pts = []

    def draw_lasso(event, x, y, flags, param):
        nonlocal drawing, pts, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            pts = [(x, y)]
            # Reset to fresh image for clean redraw
            display_img = image.copy()
            # Draw prominent starting point (dark border + bright neon yellow center)
            cv2.circle(display_img, (x, y), 6, (0, 0, 0), -1)
            cv2.circle(display_img, (x, y), 4, (0, 255, 255), -1)
            cv2.imshow(window_name, display_img)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Draw if left button is being dragged
            if drawing or (flags & cv2.EVENT_FLAG_LBUTTON and len(pts) > 0):
                drawing = True
                pts.append((x, y))
                # High-contrast dual stroke: thick black backing + vivid neon yellow line
                cv2.line(display_img, pts[-2], pts[-1], (0, 0, 0), 5, cv2.LINE_AA)
                cv2.line(display_img, pts[-2], pts[-1], (0, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow(window_name, display_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(pts) > 2:
                # Close the polygon with high-contrast line
                cv2.line(display_img, pts[-1], pts[0], (0, 0, 0), 5, cv2.LINE_AA)
                cv2.line(display_img, pts[-1], pts[0], (0, 255, 255), 3, cv2.LINE_AA)
                
                # Show subtle translucent fill to highlight selected ROI
                overlay = display_img.copy()
                poly_pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [poly_pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.25, display_img, 0.75, 0, display_img)
                cv2.polylines(display_img, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                cv2.imshow(window_name, display_img)

    window_name = "Select ROI (Click & drag freehand line, ENTER to confirm, ESC for full leaf)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_lasso)

    print("Please select a region of interest in the popup window.")
    print("Click and drag to draw a freehand shape around the leaf, then press ENTER.")

    while True:
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        
        # ENTER or SPACE to confirm
        if key == 13 or key == 32:
            break
        # ESC to cancel
        elif key == 27:
            pts = []
            break

    cv2.destroyWindow(window_name)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) > 2:
        # Fill the drawn polygon to create a mask
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    else:
        # If no valid shape drawn or canceled, return all 255
        mask.fill(255)
        
    return mask
