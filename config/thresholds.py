"""
config/thresholds.py — Central threshold configuration for the decision engine.

All numeric thresholds used by the rule-based decision engine live here.
These values are calibrated from the reference dataset (Stage 7) and must NEVER
be hardcoded inline in decision_engine.py or any other module.

Every threshold includes:
  - Its numeric value
  - A comment explaining what it represents
  - How it was derived (filled in during Stage 7 calibration)

This file is imported by src/decision_engine.py and src/feature_extraction.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Segmentation thresholds (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

# HSV ranges for leaf segmentation (used in cv2.inRange)
# These define what counts as "leaf green" vs background.
# Format: (H_min, S_min, V_min), (H_max, S_max, V_max)
LEAF_HSV_LOWER = (25, 30, 30)     # lower bound of green-ish hue range
LEAF_HSV_UPPER = (95, 255, 255)   # upper bound of green-ish hue range

# Minimum leaf area as fraction of total image area (reject if leaf is too small)
MIN_LEAF_AREA_FRACTION = 0.10

# Morphological kernel size for mask cleanup (pixels)
MORPH_KERNEL_SIZE = 7

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing thresholds (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

# Standard working resolution: longest edge resized to this many pixels
WORKING_RESOLUTION = 1024

# CLAHE parameters for brightness normalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Denoising strength (h parameter for cv2.fastNlMeansDenoisingColored)
DENOISE_H = 6
DENOISE_H_COLOR = 6

# ─────────────────────────────────────────────────────────────────────────────
# Vein extraction thresholds (Stage 5)
# ─────────────────────────────────────────────────────────────────────────────

# CLAHE for vein enhancement (typically stronger than preprocessing CLAHE)
VEIN_CLAHE_CLIP_LIMIT = 4.0
VEIN_CLAHE_TILE_GRID_SIZE = (8, 8)

# Adaptive threshold block size (must be odd) and offset constant
ADAPTIVE_THRESH_BLOCK_SIZE = 51
ADAPTIVE_THRESH_C = 10

# Frangi filter parameters (sigmas = range of vein widths to detect, in pixels)
FRANGI_SIGMAS = range(1, 6)
FRANGI_BETA1 = 0.5
FRANGI_BETA2 = 15.0
FRANGI_BLACK_RIDGES = True  # veins appear darker on backlit grayscale

# Minimum skeleton branch length (pixels) — prune shorter branches as noise
MIN_SKELETON_BRANCH_LENGTH = 15

# ─────────────────────────────────────────────────────────────────────────────
# Color / chlorosis thresholds (Stage 6 — feature extraction from front-lit)
# ─────────────────────────────────────────────────────────────────────────────

# HSV range defining "yellow / pale" pixels for yellow_pixel_ratio
YELLOW_HSV_LOWER = (15, 40, 100)
YELLOW_HSV_UPPER = (35, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# Decision engine thresholds (Stage 8 — populated after Stage 7 calibration)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: The values below are PLACEHOLDERS. They will be replaced with
# empirically calibrated values from the reference dataset during Stage 7.
# Each value includes the calibration method in its comment once set.

# Vein density (vein skeleton pixels / leaf area pixels)
# Calibration method: TBD in Stage 7 (e.g., healthy mean − 1.5×std)
VEIN_DENSITY_HEALTHY_LOW = 0.03     # placeholder — below this = possibly deficient
VEIN_DENSITY_DEFICIENT = 0.02       # placeholder — below this = deficient

# Yellow pixel ratio (fraction of leaf pixels in yellow HSV band)
# Calibration method: TBD in Stage 7
YELLOW_RATIO_POSSIBLY_DEFICIENT = 0.15   # placeholder — above this = possibly deficient
YELLOW_RATIO_DEFICIENT = 0.30            # placeholder — above this = deficient

# Excess Green Index (ExG = 2G − R − B, normalized per pixel, then averaged)
# Calibration method: TBD in Stage 7
EXG_HEALTHY_LOW = 0.10              # placeholder — below this = reduced greenness

# Dark Green Color Index (DGCI, standard turf formula)
# Calibration method: TBD in Stage 7
DGCI_HEALTHY_LOW = 0.40             # placeholder — below this = pale/chlorotic

# Mean saturation of leaf region in HSV
# Calibration method: TBD in Stage 7
MEAN_SATURATION_HEALTHY_LOW = 60    # placeholder — below this = washed out color

# Vein thickness average (pixels, from distance transform along skeleton)
# Calibration method: TBD in Stage 7
VEIN_THICKNESS_HEALTHY_LOW = 1.5    # placeholder

# Interveinal contrast (difference in mean green between on-vein and off-vein regions)
# Calibration method: TBD in Stage 7
INTERVEINAL_CONTRAST_THRESHOLD = 15.0  # placeholder — above this suggests interveinal chlorosis
