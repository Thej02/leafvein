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
# Widened to capture pale, desaturated yellow-green mottling (H up to 45, S down to 20, V down to 50)
YELLOW_HSV_LOWER = (15, 20, 50)
YELLOW_HSV_UPPER = (45, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# Decision engine thresholds (Stage 8 — populated after Stage 7 calibration)
# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY FACTORS (Color / Chlorosis)
# These factors independently trigger a NOT HEALTHY verdict.

# Yellow pixel ratio (fraction of leaf pixels in yellow HSV band)
# Calibration method: Adjusted to catch pale/mottled yellowing (>35%)
YELLOW_RATIO_DEFICIENT = 0.258292

# Excess Green Index (ExG = 2G − R − B, normalized per pixel, then averaged)
# Calibration method: Healthy mean (0.619) - 1 std (0.114)
EXG_HEALTHY_LOW = 0.632374

# Dark Green Color Index (DGCI, standard turf formula)
# Calibration method: Marginally below healthy mean (0.531)
DGCI_HEALTHY_LOW = 0.595351

# Interveinal contrast (difference in mean green between on-vein and off-vein regions)
# Calibration method: Expected to be high in Mg/Fe deficiency. Set above typical healthy baseline.
INTERVEINAL_CONTRAST_THRESHOLD = 62.692146

# Color spatial variance (confound check for localized damage vs systemic deficiency)
# Calibration method: Adjusted to catch high variance mottling (e.g., >500)
COLOR_SPATIAL_VARIANCE_MAX = 676.898837

# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY FACTORS (Vein Geometry)
# ─────────────────────────────────────────────────────────────────────────────
# These factors do NOT trigger a NOT HEALTHY verdict on their own. They are
# appended as supporting evidence if a primary color factor fails.

# Vein density (vein pixels / leaf area)
# Calibration method: Healthy mean (0.048) - 1 std (0.011)
VEIN_DENSITY_DEFICIENT = 0.050208

# Vein thickness (average pixel width)
# Calibration method: Healthy mean (7.49) + slightly below 1 std (0.68)
VEIN_THICKNESS_DEFICIENT_HIGH = 7.653426

# ─────────────────────────────────────────────────────────────────────────────
# Reporting thresholds (Stage 10)
# ─────────────────────────────────────────────────────────────────────────────

# Minimum area (in pixels) for a blob of flagged pixels to be circled
# Drops single-pixel noise from the "unhealthy regions" visualization
UNHEALTHY_REGION_MIN_AREA = 250
