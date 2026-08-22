# IMPLEMENTATION_PLAN.md

> Living document. This plan **implements** `goal.md` and must never contradict it.
> Every time this plan changes (a step is added, reordered, reworked, or a design
> decision is revised), log it in `changes.md` with date, what changed, and why —
> **before or immediately after** making the edit here. Do not silently edit this file.

## 0. Guardrails for the coding agent (read first, every session)
- No ML libraries as classifiers: do not import/use `sklearn.*Classifier`,
  `tensorflow`, `torch`, `keras`, or write any train/fit/predict pipeline for the
  health decision itself. `scikit-learn` may only be used for non-learned utilities
  if truly needed (e.g., `StandardScaler` for reporting z-scores) — prefer plain
  NumPy even for that.
- All decision thresholds must live in one clearly named, commented config
  (`config/thresholds.yaml` or `config/thresholds.py`) — never hardcoded inline
  scattered across files.
- Every function that outputs a verdict must also output the *reasoning* (which
  features tripped which thresholds) — this is a product requirement, not optional
  logging.
- Before writing code for a new stage, re-read the relevant section below in full.
- After finishing each stage, update `changes.md` (see template at bottom) and check
  the corresponding box in `goal.md`'s Success Criteria if applicable — but only
  check a goal.md box, never edit its text.

---

## Stage 1 — Project Scaffolding
**Deliverable:** repo skeleton, environment, config.

```
project/
├── goal.md
├── implementation_plan.md
├── changes.md
├── requirements.txt
├── config/
│   └── thresholds.py
├── data/
│   ├── raw/              # original captured images, unmodified
│   ├── reference/        # calibration subset (healthy + deficient)
│   ├── validation/       # held-out subset for testing rules
│   └── labels.csv        # image_id, label, ground_truth_basis, notes
├── src/
│   ├── capture_protocol.md
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── vein_extraction.py
│   ├── feature_extraction.py
│   ├── decision_engine.py
│   ├── report_generator.py
│   └── pipeline.py        # orchestrates the above end-to-end
├── cli.py                 # entry point: `python cli.py --image path.jpg`
├── notebooks/
│   └── calibration_exploration.ipynb   # for threshold-setting only, not production code
└── tests/
    ├── test_segmentation.py
    ├── test_vein_extraction.py
    ├── test_feature_extraction.py
    └── test_decision_engine.py
```

**Tasks:**
1. Initialize git repo, `.gitignore` (exclude raw large images if needed, keep small
   sample set for tests).
2. `requirements.txt`: `opencv-python`, `numpy`, `scikit-image` (for skeletonization —
   `skimage.morphology.skeletonize`, this is classical image processing, not ML),
   `matplotlib` (for calibration plots), `pyyaml`, `pytest`. No `torch`/`tensorflow`.
3. Create empty `config/thresholds.py` with placeholder structure (filled in Stage 5).
4. Create `data/labels.csv` header: `image_id,capture_date,label,ground_truth_basis,notes`.

---

## Stage 2 — Capture Protocol (do this before collecting any data)
**Deliverable:** `src/capture_protocol.md` — a document, not code, but critical.

**Tasks:**
1. Specify **backlighting setup**: leaf placed flat between two sheets of frosted/
   diffusing material (e.g., tracing paper) or directly against a lightbox / phone
   flashlight through a diffuser, photographed from directly above.
2. Specify: fixed camera distance (e.g., 15 cm), leaf fully filling ~60–80% of frame,
   plain contrasting background visible around the leaf edges (for segmentation),
   camera in manual/locked exposure if possible to avoid auto-brightness drift
   between shots.
3. Specify minimum resolution (e.g., ≥ 8MP, matches goal.md hardware requirement).
4. Specify: capture 2 images per leaf — one **backlit** (for vein extraction) and one
   **normal front-lit, diffuse daylight** (for color/chlorosis analysis) — same leaf,
   same session. Both are needed; they serve different feature-extraction stages.
5. Document how each leaf's ground-truth label is assigned (visual grading against a
   reference symptom chart by whoever is closest to "expert" available — project
   guide, or documented published symptom-chart comparison). Record this in
   `labels.csv` per `goal.md` constraint #7.

---

## Stage 3 — Data Collection
**Deliverable:** ≥60 leaf image pairs in `data/raw/`, `data/labels.csv` populated.

**Tasks:**
1. Collect target: ~30 healthy, ~30 showing some degree of discoloration/suspected
   deficiency (mixed severity is fine and useful).
2. Split roughly 70/30 into `data/reference/` (calibration) and `data/validation/`
   (held-out) — do this by copying/symlinking, keep `raw/` as the untouched archive.
3. Sanity-check every image manually against the capture protocol before including it
   (reject blurry, off-center, or improperly lit captures — log rejections in
   `changes.md` under a "data notes" entry, not silently discarded).

---

## Stage 4 — Preprocessing & Segmentation (`preprocessing.py`, `segmentation.py`)
**Deliverable:** function that takes a raw image → clean, background-removed,
normalized leaf image.

**Tasks:**
1. `preprocessing.py`:
   - Resize to a standard working resolution (e.g., long edge = 1024px) for
     consistent downstream measurements.
   - Denoise (`cv2.fastNlMeansDenoisingColored` or Gaussian blur) lightly — don't
     blur away fine vein detail.
   - Normalize brightness/contrast (CLAHE on the L channel in LAB color space is a
     good classical choice, no learning involved).
2. `segmentation.py`:
   - Convert to HSV; threshold on hue/saturation to separate leaf-green(ish) region
     from background (use `cv2.inRange` with a tunable range, exposed in config).
   - Clean mask with morphological open/close (`cv2.morphologyEx`).
   - Take largest connected component (`cv2.connectedComponentsWithStats`) as the
     leaf region; discard small noise blobs.
   - Output: binary leaf mask + masked leaf image, plus **leaf area in pixels**
     (needed later to normalize vein density).
3. Unit test: run on 3–5 known sample images, assert mask covers a sane % of frame
   and has one connected component.

---

## Stage 5 — Vein Extraction (`vein_extraction.py`)
**Deliverable:** binary vein skeleton image + vein pixel count, from the **backlit**
image only.

**Tasks:**
1. Apply the leaf mask from Stage 4 to the backlit image.
2. Enhance vein contrast: convert to grayscale, apply CLAHE, then a ridge/edge
   enhancement — options to try and pick empirically:
   - Adaptive thresholding (`cv2.adaptiveThreshold`) tuned so veins (locally darker
     or lighter than surrounding tissue depending on backlighting) come out as a
     connected binary structure.
   - Frangi vesselness filter (`skimage.filters.frangi`) — designed for exactly this
     kind of thin curvilinear structure detection (originally for blood vessels;
     works well on leaf veins), still classical, not ML.
3. Skeletonize the binary vein mask (`skimage.morphology.skeletonize`) → 1-pixel-wide
   vein network.
4. Remove tiny spurious branches (prune skeleton components below a pixel-length
   threshold) to reduce noise from segmentation artifacts.
5. Output artifacts to save per image: vein skeleton mask, vein pixel count,
   skeleton branch-point count (proxy for branching complexity, optional secondary
   feature).
6. Unit test + **visual sanity check step**: save overlay images (skeleton drawn in
   red over original leaf) for every calibration image into
   `data/reference/_debug_overlays/` so the person can visually confirm veins are
   actually being traced correctly before trusting any numbers from this stage.
   This visual QA step is mandatory — do not proceed to Stage 6 thresholds until
   overlays look right on at least 90% of the reference set.

---

## Stage 6 — Feature Extraction (`feature_extraction.py`)
**Deliverable:** a feature dict per leaf, from both images.

**From backlit image (vein features):**
- `vein_density` = vein skeleton pixel count / leaf area (pixels) — the core metric.
- `vein_thickness_avg` = estimate via distance transform on the pre-skeleton binary
  vein mask, averaged along the skeleton (`cv2.distanceTransform` then sample along
  skeleton coordinates).
- `branch_point_count` (optional secondary signal).

**From front-lit image (color/chlorosis features):**
- Convert masked leaf region to HSV and LAB.
- `mean_hue`, `mean_saturation` — overall greenness proxy.
- `yellow_pixel_ratio` = fraction of leaf-mask pixels falling in a defined
  "yellow/pale" HSV band (config-defined) — direct chlorosis proxy.
- `excess_green_index` (ExG = 2G − R − B, normalized) — a well-established vegetation
  color index, computed leaf-pixel-wise then averaged.
- `dark_green_color_index` (DGCI) — standard formula from turf/plant-health
  literature, also purely arithmetic on HSB values.
- Optional: interveinal vs. on-vein color contrast — sample color near skeleton
  pixels vs. away from them, to catch "veins stay green, tissue yellows" patterns
  (classic Mg/Fe deficiency signature) as a distinct feature from overall yellowing.

**Tasks:**
1. Implement each as a small, independently testable function.
2. Store all features per image into a single `data/features.csv`
   (image_id + all feature columns + label from labels.csv, joined).
3. Unit test each feature function on 1–2 synthetic/known images with expected
   value ranges.

---

## Stage 7 — Threshold Calibration (exploratory, in the notebook — not production code)
**Deliverable:** documented threshold values written into `config/thresholds.py`.

**Tasks:**
1. In `notebooks/calibration_exploration.ipynb`, load `data/features.csv` for the
   **reference set only** (never touch validation set here — that would contaminate
   the held-out test).
2. For each feature, plot distributions split by label (healthy vs. deficient) —
   histograms or box plots.
3. For each feature, pick a threshold/band using a simple, explainable statistical
   rule — e.g., "healthy mean ± 1.5×std" as the normal band, or the midpoint between
   healthy-mean and deficient-mean if separation is clean. Document the chosen method
   and the resulting numeric values in `config/thresholds.py` comments, and mirror
   the reasoning in `changes.md`.
4. This is the ONLY place "learning from data" happens, and it is explicitly a
   transparent statistical calibration of human-written rules — not a trained model,
   and it must stay documented and inspectable (this distinction matters for the
   viva — be ready to explain it exactly this way).

---

## Stage 8 — Decision Engine (`decision_engine.py`)
**Deliverable:** deterministic function `evaluate(features: dict) -> Verdict`.

**Tasks:**
1. Implement explicit if-then rules using the thresholds from Stage 7, e.g.:
   ```
   if vein_density < threshold_low OR yellow_pixel_ratio > threshold_high:
       flag "Possibly Deficient"
   if (vein_density < threshold_very_low) AND (yellow_pixel_ratio > threshold_very_high):
       flag "Deficient"
   else:
       "Healthy"
   ```
   (Actual structure to be refined during Stage 7 based on what separates the classes
   well — keep it simple and inspectable; a small decision tree you *wrote*, not one
   that was *fit*, is fine and still within the no-ML constraint.)
2. Attach a **reason string** to every verdict listing which specific features
   crossed which thresholds and what deficiency pattern that combination resembles
   (per the Reference Vocabulary in `goal.md`) — with appropriate hedging language
   ("pattern consistent with…", not a diagnostic claim).
3. Unit tests with hand-constructed feature dicts covering each branch of the logic.

---

## Stage 9 — Validation
**Deliverable:** accuracy/precision/recall + confusion matrix of the rule engine
against `data/validation/`.

**Tasks:**
1. Run the full pipeline end-to-end on every validation image (never used in Stage 7).
2. Compare predicted verdict vs. ground-truth label from `labels.csv`.
3. Report confusion matrix, precision/recall/F1 per class, and manually review every
   misclassified case — note in `changes.md` / final report whether it's a capture
   issue, a feature-extraction issue, or a genuinely ambiguous borderline leaf.
4. Explicitly frame this in the report as "rule-engine validation accuracy," not
   "model accuracy," to stay consistent with `goal.md`.

---

## Stage 10 — Report Generator & Interface (`report_generator.py`, `cli.py`)
**Deliverable:** end-user-facing output.

**Tasks:**
1. `report_generator.py`: takes verdict + reasoning + feature values → formats a
   plain-language report (plus optionally a saved annotated image: leaf with vein
   overlay and a text panel).
2. `cli.py`: `python cli.py --backlit path1.jpg --frontlit path2.jpg` → runs full
   pipeline → prints/saves report.
3. **Stretch goal** (only after CLI works end-to-end and Stage 9 validation is done):
   simple Streamlit or Flask single-page upload-and-report UI, per `goal.md`'s
   "stretch goal" note. Do not start this before the CLI + validation are solid.

---

## Stage 11 — Documentation & Phase-II Report Writeup
**Tasks:**
1. Write up methodology, calibration reasoning, validation results, and limitations
   (dataset size, single-species scope, capture-protocol sensitivity) for the formal
   project report, mirroring `goal.md` scope exactly.
2. Fix the known literature-review defect flagged during Phase-I review: Article
   2.12.2 in the existing report ("LSTM and reinforcement learning... sensor
   data...") is mismatched content unrelated to the cited tomato leaf disease paper —
   correct it to actually describe that paper's CNN-based methodology before
   Phase-II submission.
3. Update the abstract/system description to explicitly state "rule-based /
   classical image processing, not machine learning" wherever the current Phase-I
   report says "AI-ML model" — keep the two documents (report vs. actual system)
   consistent.

---

## Suggested Order of Execution
Stage 1 → 2 → 3 (data collection can run in parallel with early coding) → 4 → 5
(with mandatory visual QA gate) → 6 → 7 → 8 → 9 → 10 → 11. Do not skip the Stage 5
visual QA gate or the Stage 7/9 train/validation separation — both are common places
this kind of project silently produces meaningless numbers.

---

## Change Log Reminder
Every time any decision in this plan is added, changed, or dropped — including
threshold values chosen in Stage 7, any deviation from the folder structure, any
feature that got dropped because it didn't separate classes well, or any capture
protocol adjustment discovered during real data collection — add an entry to
`changes.md` using its template. `goal.md` itself is never edited.
