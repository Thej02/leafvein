import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import resize_to_working_resolution, denoise, normalize_brightness
from src.segmentation import segment_leaf
from config.thresholds import YELLOW_HSV_LOWER, YELLOW_HSV_UPPER

img_path = 'data/reference/3.jpeg'
img = cv2.imread(img_path)
img_resized = resize_to_working_resolution(img)
img_preprocessed = normalize_brightness(denoise(img_resized))

seg = segment_leaf(img_preprocessed)
mask = seg['mask']

hsv = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2HSV)

# Extract pixels within leaf
h = hsv[:,:,0][mask > 0]
s = hsv[:,:,1][mask > 0]
v = hsv[:,:,2][mask > 0]

print(f"Current YELLOW bounds: {YELLOW_HSV_LOWER} to {YELLOW_HSV_UPPER}")
print(f"Hue: min={h.min()}, max={h.max()}, mean={h.mean():.1f}, median={np.median(h)}")
print(f"Sat: min={s.min()}, max={s.max()}, mean={s.mean():.1f}, median={np.median(s)}")
print(f"Val: min={v.min()}, max={v.max()}, mean={v.mean():.1f}, median={np.median(v)}")

# The user says the leaf is 40-50% pale yellow-green.
# Let's find the 50th percentile of hue (which is median)
# and let's see the HSV of the 20% "yellowest/palest" pixels.
# Pale/yellow-green usually means lower Hue (closer to 20-40 instead of 60) and lower Saturation.

# Let's sort by Hue ascending (yellow is lower hue than green in OpenCV, green is ~60, yellow is ~30)
sorted_h_indices = np.argsort(h)
pale_h = h[sorted_h_indices[:len(h)//2]]
pale_s = s[sorted_h_indices[:len(s)//2]]
pale_v = v[sorted_h_indices[:len(v)//2]]

print(f"\nStats for the 'yellower' 50% of the leaf (lowest Hue):")
print(f"Hue: min={pale_h.min()}, max={pale_h.max()}, mean={pale_h.mean():.1f}")
print(f"Sat: min={pale_s.min()}, max={pale_s.max()}, mean={pale_s.mean():.1f}")
print(f"Val: min={pale_v.min()}, max={pale_v.max()}, mean={pale_v.mean():.1f}")

# Calculate how much falls in current threshold
current_mask = cv2.inRange(hsv, np.array(YELLOW_HSV_LOWER), np.array(YELLOW_HSV_UPPER))
current_ratio = cv2.countNonZero(cv2.bitwise_and(current_mask, mask)) / cv2.countNonZero(mask)
print(f"\nCurrent ratio: {current_ratio:.2%}")

# Let's test a wider threshold
test_lower = (20, 20, 50)
test_upper = (45, 255, 255)
test_mask = cv2.inRange(hsv, np.array(test_lower), np.array(test_upper))
test_ratio = cv2.countNonZero(cv2.bitwise_and(test_mask, mask)) / cv2.countNonZero(mask)
print(f"Ratio with (20, 20, 50) to (45, 255, 255): {test_ratio:.2%}")

test_lower2 = (15, 20, 50)
test_upper2 = (50, 255, 255)
test_mask2 = cv2.inRange(hsv, np.array(test_lower2), np.array(test_upper2))
test_ratio2 = cv2.countNonZero(cv2.bitwise_and(test_mask2, mask)) / cv2.countNonZero(mask)
print(f"Ratio with (15, 20, 50) to (50, 255, 255): {test_ratio2:.2%}")

