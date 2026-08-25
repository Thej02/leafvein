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

## Health-Verification Factors Correction (2026-08-25)
1. **Added `color_spatial_variance`**: Implemented in `src/feature_extraction.py` as a confound check (distinguishes systemic deficiency from localized patches).
2. **Primary vs. Secondary Flag Logic**: Updated `src/decision_engine.py` to differentiate between primary factors (color/chlorosis) and secondary factors (vein geometry). A NOT HEALTHY verdict now requires at least one primary factor failure.
3. **Threshold Calibration**: Recalibrated values in `config/thresholds.py` based on reference data means and standard deviations.
4. **Regression Tests**: Rewrote `tests/test_decision_engine.py` to match the new rules and added a test to ensure secondary-only failures do not trigger a NOT HEALTHY verdict.
5. **False Negative Bug Fix (3.jpeg)**: 
   - *Issue*: `data/reference/3.jpeg` is heavily mottled/speckled (visually ~40-50% pale yellow-green area) but returned HEALTHY with a `yellow_pixel_ratio` of only 1.24%. The narrow HSV band was completely missing the desaturated yellow-green mottling.
   - *Fix*: Widened the yellow HSV band in `config/thresholds.py` from `(15, 40, 100)` -> `(35, 255, 255)` to a more inclusive `(15, 20, 50)` -> `(45, 255, 255)`. 
   - *Result*: The `yellow_pixel_ratio` on `3.jpeg` correctly jumped to **39.11%**, exceeding the calibrated failure threshold. The `color_spatial_variance` factor also correctly triggered at **522.4**, flagging the leaf as NOT HEALTHY due to a potential pest/fungal issue rather than systemic deficiency. The output report was also updated to print the pass/fail thresholds next to measured values for easier debugging.
