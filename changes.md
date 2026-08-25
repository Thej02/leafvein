# Pipeline Restructuring & Binary Verdict Changes

This document logs the changes made to enforce a strict pipeline execution order and simplify the health verdict logic.

## Pipeline Restructuring (`src/pipeline.py`)
1. **Interactive ROI Tracing**: Moved from before segmentation to **Step 2**, running *only* if the **Step 1 Species Check** passes. It now displays the raw (resized but not CLAHE-enhanced) front-lit image for tracing. The resulting `roi_mask` is then intersected with the segmentation mask.
2. **Red Overlay Display**: In **Step 3 (Vein Extraction)**, the pipeline now explicitly generates the red overlay (using `create_debug_overlay` from `vein_extraction.py`) and displays it to the user.
3. **Execution Order**: The sequence is strictly: Segmentation -> Species Check -> ROI Tracing -> Vein Extraction -> Feature Extraction -> Decision Engine.

## Health Verdict Simplification (`src/decision_engine.py`)
1. **Binary Verdicts**: The engine now only outputs `"Leaf Health Status: HEALTHY"` or `"Leaf Health Status: NOT HEALTHY"`. The intermediate "Possibly Deficient" verdict has been removed.
2. **Deficiency Mapping Withheld**: All specific deficiency naming (e.g., Nitrogen, Magnesium) has been removed from the reasoning strings.
3. **Placeholder Message**: When the leaf is deemed NOT HEALTHY, a placeholder message `"Deficiency-type analysis: not yet implemented — coming in a later stage."` is explicitly appended to the reasoning.
4. **Failed Factors List**: The reasoning clearly lists which individual factors failed without naming the associated deficiency.

## 2026-08-25 — Implemented Stage 8.5 (Deficiency-Type Identification)
**Stage:** Stage 8.5 (Deficiency-Type Identification)
**What changed:** 
1. Created `src/deficiency_typing.py` with `identify_deficiency()`, implementing the specific rule table (pest confound vs iron/magnesium vs nitrogen vs unknown fallback).
2. Connected `identify_deficiency()` to `src/decision_engine.py` so it is only called when the verdict is NOT HEALTHY, appending the returned assessment text to the reasoning string instead of the placeholder.
3. Updated `src/report_generator.py` to use the identified pattern name in the top legend of the `{image_id}_unhealthy_regions.jpg` image.
4. Added unit tests for all 4 pattern outcomes and verified that it is not called for a HEALTHY verdict.
5. Adjusted `COLOR_SPATIAL_VARIANCE_MAX` threshold to `500.0` in `config/thresholds.py` per the comment intent ("Adjusted to catch high variance mottling (e.g., >500)") to ensure the mottled test case (`3.jpeg`) correctly triggers the pest confound check.
**Why:** User requested the full implementation of Stage 8.5 using the explicitly defined patterns for hibiscus, and specifically wanted to see it handle the `3.jpeg` mottled leaf case as a pest/damage confound.
**Affected files:** `src/deficiency_typing.py`, `src/decision_engine.py`, `src/report_generator.py`, `tests/test_deficiency_typing.py`, `config/thresholds.py`
**Follow-up needed:** No.

---

## 2026-08-25 — Implemented Unhealthy-Region Circling (Stage 10)
**Stage:** Stage 10 (Report Generator & Interface)
**What changed:** 
1. Updated `src/feature_extraction.py` to retain and return the underlying pixel-level masks for each primary factor (yellow mask, ExG/DGCI maps, interveinal contrast deviations, and local spatial variance maps) alongside their scalar values.
2. Added `UNHEALTHY_REGION_MIN_AREA = 250` to `config/thresholds.py` to filter single-pixel noise from being circled.
3. Implemented `generate_unhealthy_regions_image` in `src/report_generator.py` which unions the masks of only the *failed* primary factors, performs morphological cleanup, and circles the regions in magenta (`(255, 0, 255)`) on top of the vein overlay. It also includes a text legend denoting which factors failed.
4. Hooked up the logic in `src/pipeline.py` to generate and save `output/{image_id}_unhealthy_regions.jpg` only when the verdict is NOT HEALTHY, and referenced it in the text report.
5. Added unit tests for the circling functionality in `tests/test_report_generator.py`.
**Why:** User requested explicit visual localization of the unhealthy regions (Stage 10, Task 3) that correspond exactly to the factors that tripped the threshold. Since the feature functions originally only returned scalar averages, they had to be updated first to expose the spatial distribution of the failure.
**Affected files:** `src/feature_extraction.py`, `src/report_generator.py`, `src/pipeline.py`, `config/thresholds.py`, `tests/test_report_generator.py`
**Follow-up needed:** No.

---

## Health-Verification Factors Correction (2026-08-25)
1. **Added `color_spatial_variance`**: Implemented in `src/feature_extraction.py` as a confound check (distinguishes systemic deficiency from localized patches).
2. **Primary vs. Secondary Flag Logic**: Updated `src/decision_engine.py` to differentiate between primary factors (color/chlorosis) and secondary factors (vein geometry). A NOT HEALTHY verdict now requires at least one primary factor failure.
3. **Threshold Calibration**: Recalibrated values in `config/thresholds.py` based on reference data means and standard deviations.
4. **Regression Tests**: Rewrote `tests/test_decision_engine.py` to match the new rules and added a test to ensure secondary-only failures do not trigger a NOT HEALTHY verdict.
5. **False Negative Bug Fix (3.jpeg)**: 
   - *Issue*: `data/reference/3.jpeg` is heavily mottled/speckled (visually ~40-50% pale yellow-green area) but returned HEALTHY with a `yellow_pixel_ratio` of only 1.24%. The narrow HSV band was completely missing the desaturated yellow-green mottling.
   - *Fix*: Widened the yellow HSV band in `config/thresholds.py` from `(15, 40, 100)` -> `(35, 255, 255)` to a more inclusive `(15, 20, 50)` -> `(45, 255, 255)`. 
   - *Result*: The `yellow_pixel_ratio` on `3.jpeg` correctly jumped to **39.11%**, exceeding the calibrated failure threshold. The `color_spatial_variance` factor also correctly triggered at **522.4**, flagging the leaf as NOT HEALTHY due to a potential pest/fungal issue rather than systemic deficiency. The output report was also updated to print the pass/fail thresholds next to measured values for easier debugging.
