# Capture Protocol for *Rosa-sinensis* Leaf Imaging

> This document defines the standardized procedure for capturing leaf images
> suitable for the vein-extraction and color-analysis pipeline. Follow every step
> exactly — the downstream processing is calibrated to these conditions.

## Equipment Required
- Smartphone camera, ≥ 8 MP (most modern phones qualify)
- A lightbox, tablet screen at max white brightness, or phone flashlight placed
  behind a sheet of frosted/translucent material (tracing paper, baking parchment,
  or a thin white plastic diffuser)
- A flat, non-reflective, contrasting background (black or dark-colored card)
  visible around the leaf edges (needed for segmentation)
- Optional: a small stand or stack of books to hold the camera at a fixed distance

## Two-Image Protocol (both required per leaf)

### Image 1 — Backlit / Transmitted Light (for vein extraction)
1. Place the diffuser material flat on a surface.
2. Place the lightbox / flashlight / white-screen tablet beneath the diffuser,
   pointing upward.
3. Lay the leaf flat on top of the diffuser, adaxial (top) surface facing the camera.
   Ensure the leaf is fully unfolded — no curling, no overlapping edges.
4. Place the dark contrasting background around the leaf (or use a frame/cutout)
   so that the leaf edges are clearly defined against a dark surround.
5. Position the camera directly above the leaf (perpendicular to the surface),
   at approximately **15 cm distance** from the leaf surface.
6. The leaf should fill approximately **60–80%** of the frame.
7. **Lock exposure** if possible (tap-and-hold on the leaf in the camera app to
   lock AE/AF). If the camera doesn't support locking, keep the framing consistent
   across all shots to minimize auto-brightness drift.
8. Capture the image. The veins should appear as darker lines against the
   translucent (brighter) tissue.

### Image 2 — Front-lit / Diffuse Daylight (for color/chlorosis analysis)
1. Remove the backlight. Place the same leaf flat on the dark background.
2. Illuminate with **diffuse natural daylight** (e.g., near a window but not in
   direct sunlight) or a diffused artificial white light from above and slightly
   to the side. Avoid harsh shadows.
3. Same camera distance (~15 cm), same framing (leaf fills 60–80% of frame).
4. Lock exposure if possible.
5. Capture the image. The goal is accurate color representation — the image should
   look like what the leaf looks like to the naked eye.

## File Naming Convention
Each leaf gets a unique ID (e.g., `leaf_001`). Files are named:
- `leaf_001_backlit.jpg` — Image 1
- `leaf_001_frontlit.jpg` — Image 2

Both files go into `data/raw/`.

## Minimum Resolution
≥ 8 MP (approximately 3264 × 2448 or higher). Most smartphone cameras in 2024+
exceed this. Do NOT crop or resize before saving to `raw/` — preprocessing handles
resizing programmatically.

## Quality Checklist (reject and re-shoot if any fail)
- [ ] Leaf is flat, fully unfolded, no curling edges
- [ ] Leaf fills 60–80% of frame, not cut off at edges
- [ ] Background contrast is clear (dark surround visible around all leaf edges)
- [ ] Image is in focus (veins should be sharp, especially in backlit image)
- [ ] No harsh shadows or specular reflections on the leaf surface
- [ ] Backlit image: veins are visibly discernible as darker/lighter lines
- [ ] Front-lit image: color looks natural (not overexposed or underexposed)
- [ ] Camera was approximately perpendicular to the leaf (no strong perspective distortion)

## Ground Truth Labeling
For each leaf, record in `data/labels.csv`:
1. `image_id`: e.g., `leaf_001`
2. `capture_date`: ISO date of capture
3. `label`: one of `healthy`, `possibly_deficient`, `deficient`
4. `ground_truth_basis`: how the label was determined. Acceptable methods:
   - "Visual grading against [reference chart name/source]"
   - "Expert assessment by [name/role]"
   - "Lab soil/tissue test results: [summary]" (if available)
5. `notes`: any observations (e.g., "slight yellowing on older leaves",
   "veins appear thinner than typical healthy specimen")

## Important Notes
- Capture both images in the **same session** — do not photograph the backlit image
  one day and the front-lit image another day (the leaf's condition may change).
- Handle leaves gently; do not damage them if the intent is to track the same plant
  over time.
- If working outdoors, be consistent about time of day for the front-lit images
  (color temperature of daylight changes between morning and afternoon).
