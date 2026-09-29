# Rosa-sinensis Leaf Health Severity Classification Engine

This module introduces **multi-paradigm severity classification** and **actionable botanical recommendations** for *Hibiscus rosa-sinensis* leaf health. It compares three distinct computational approaches on the exact same dataset:

1. **Rule-Based Decision Engine** (`severityClass/rule_based.py`) — A transparent, deterministic botanical scoring model combining chlorosis extent, vascular contrast, vein density deviation, and color indices with zero training required.
2. **Classical Machine Learning** (`severityClass/ml_model.py`) — Random Forest and Support Vector Machine (SVM) classifiers trained on the calibrated 9-feature botanical vector with Leave-One-Out Cross-Validation (LOOCV).
3. **Deep Learning Transfer Learning & Grad-CAM** (`severityClass/dl_model.py`) — MobileNetV2 fine-tuned with horticultural image augmentation, evaluated with confusion matrices and Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the spatial regions driving predictions.
4. **Actionable Recommendations** (`severityClass/recommendations.py`) — Botanical diagnostic lookups mapping each severity grade to specific nutritional and cultural interventions.
5. **Multi-Model Benchmark Engine** (`severityClass/compare.py`) — Side-by-side consensus evaluation comparing agreement rates, accuracy against reference ground truth, and per-leaf recommendations.

---

## 1. Rule-Based Severity Scoring Formula

### Mathematical Formulation
The rule-based severity score $S \in [0, 100]$ evaluates multi-spectral chlorophyll degradation and vascular architecture disruption:

$$S_{\text{total}} = \min\left(100.0, \; \max\left(0.0, \; w_y S_{\text{yellow}} + w_c S_{\text{contrast}} + w_d S_{\text{density}} + w_g S_{\text{dgci}} + P_{\text{confound}}\right)\right)$$

Where the component weights are:
- $w_y = 0.35$ (Interveinal Yellowing Extent)
- $w_c = 0.25$ (Vein-to-Interveinal Contrast)
- $w_d = 0.20$ (Vascular Density Deficit)
- $w_g = 0.20$ (Dark Green Color Index Loss)

$$\sum w_i = 1.00$$

### Component Definitions & Botanical Rationale

1. **Interveinal Yellowing Extent ($S_{\text{yellow}}$)**:
   $$S_{\text{yellow}} = \min\left(100.0, \; \frac{\text{yellow\_pixel\_ratio}}{0.60} \times 100.0\right)$$
   *Rationale*: Chlorosis is the most direct indicator of chlorophyll breakdown. A yellow pixel ratio exceeding 60% indicates extensive chloroplast collapse across the leaf lamina.

2. **Vein-to-Interveinal Contrast ($S_{\text{contrast}}$)**:
   $$S_{\text{contrast}} = \begin{cases} 0.0 & \text{if } C \le 25.0 \\ \min\left(100.0, \; \frac{C - 25.0}{40.0} \times 100.0\right) & \text{if } C > 25.0 \end{cases}$$
   *Rationale*: Interveinal chlorosis (the hallmark symptom of magnesium and iron deficiency) creates high chromatic contrast ($C > 40.0$) between dark green veins and chlorotic intervening tissue. Uniform green leaves exhibit baseline contrast $C \le 25.0$.

3. **Vascular Density Deficit ($S_{\text{density}}$)**:
   $$S_{\text{density}} = \begin{cases} 0.0 & \text{if } D \ge D_{\text{base}} \\ \min\left(100.0, \; \frac{D_{\text{base}} - D}{D_{\text{base}}} \times 100.0\right) & \text{if } D < D_{\text{base}} \end{cases}$$
   where $D_{\text{base}} = 0.0550$ (calibrated healthy mean).
   *Rationale*: Chronic deficiency or developmental stress stunts minor vein morphogenesis, reducing the total skeletonized vein length per unit leaf area.

4. **Dark Green Color Index Loss ($S_{\text{dgci}}$)**:
   $$S_{\text{dgci}} = \begin{cases} 0.0 & \text{if } \text{DGCI} \ge 0.650 \\ \min\left(100.0, \; \frac{0.650 - \text{DGCI}}{0.350} \times 100.0\right) & \text{if } \text{DGCI} < 0.650 \end{cases}$$
   *Rationale*: DGCI is an established agricultural index combining Hue, Saturation, and Brightness that directly correlates with total foliar nitrogen and chlorophyll content.

5. **Confound Spatial Variance Penalty ($P_{\text{confound}}$)**:
   If `color_spatial_variance` exceeds the calibrated threshold of $500.0$:
   $$P_{\text{confound}} = \min\left(10.0, \; \frac{V - 500.0}{500.0} \times 10.0\right)$$
   *Rationale*: High spatial color variance indicates sharp, localized mottling or pest feeding stipples (e.g., spider mites) rather than smooth systemic chlorosis.

---

## 2. Severity Classification Thresholds

The continuous score $S_{\text{total}} \in [0, 100]$ is partitioned into four distinct tiers:

| Severity Tier | Score Range | Botanical State |
|---|---|---|
| **Healthy** | $0.0 \le S < 20.0$ | Uniform deep green lamina ($\text{DGCI} \ge 0.65$), minimal chlorosis ($<5\%$), intact vein density ($\ge 0.05$). |
| **Mild** | $20.0 \le S < 45.0$ | Incipient stress; localized subtle chlorosis ($5\% - 25\%$) or minor interveinal contrast divergence. Intact vascular network. |
| **Moderate** | $45.0 \le S < 70.0$ | Established nutrient deficiency; pronounced interveinal chlorosis ($25\% - 50\%$) or noticeable vein density deficit. |
| **Severe** | $S \ge 70.0$ | Advanced chlorosis ($>50\%$), severe vascular degradation, or bleached lamina with impending necrosis and leaf drop. |

---

## 3. Actionable Botanical Recommendations

| Severity Level | Primary Botanical Diagnosis | Actionable Interventions | Target Nutrients / Cultural Fix |
|---|---|---|---|
| **Healthy** | Optimal Plant Vigor | Maintain deep root irrigation; ensure 6+ hours direct sunlight; monthly balanced feeding (7-2-12 or 10-10-10). | Balanced NPK + trace minerals (Maintenance) |
| **Mild** | Incipient Nutrient Chlorosis / pH Lock | Allow top 2 inches soil to dry between waterings; check soil pH (ideal 6.0–6.8); apply dilute foliar iron + magnesium sulfate (Epsom salt). | Foliar Iron (Fe) & Magnesium (Mg); pH correction |
| **Moderate** | Established Fe/Mg Chlorosis | Drench root zone with chelated iron (Fe-EDDHA); spray Epsom salt (1 tbsp/gal); feed hibiscus fertilizer with elevated potassium; inspect leaf undersides for mites. | Chelated Iron (Fe), Magnesium (Mg), Potassium (K) |
| **Severe** | Critical Chlorosis & Vascular Breakdown | Flush soil with fresh water to clear fertilizer salt toxicity; administer comprehensive chelated micro-pack (Fe, Mg, Mn, Zn) foliar feed; prune necrotic foliage. | Emergency full-spectrum micronutrients + root flush |

---

## 4. How to Run Each Script

### A. Rule-Based Severity Evaluator
Analyze a single leaf image:
```bash
python -m severityClass.rule_based --image sample_images/1.jpeg
```
With annotated image export:
```bash
python -m severityClass.rule_based --image sample_images/2.jpeg --output output/
```

### B. Classical Machine Learning (Random Forest & SVM)
Extract features across the dataset, train models, and compute LOOCV metrics:
```bash
python -m severityClass.ml_model --train
```
Predict severity for a single leaf using a trained model:
```bash
python -m severityClass.ml_model --predict sample_images/1.jpeg --model rf
python -m severityClass.ml_model --predict sample_images/2.jpeg --model svm
```

### C. Deep Learning (MobileNetV2 Transfer Learning & Grad-CAM)
Train MobileNetV2 and generate Grad-CAM heatmaps for representative classes:
```bash
python -m severityClass.dl_model --train --epochs 25
```
Run single-image prediction with Grad-CAM visualization:
```bash
python -m severityClass.dl_model --predict sample_images/2.jpeg --gradcam
```

### D. Benchmark & Multi-Methodology Comparison
Run all three paradigms side-by-side across the dataset:
```bash
python -m severityClass.compare
```
Exports `severityClass/comparison_table.csv` and `severityClass/comparison_summary.txt`.

---

## 5. Directory Structure (`severityClass/`)

```
severityClass/
├── __init__.py                # Package exports
├── recommendations.py        # Horticultural lookup table & guidance
├── rule_based.py              # Multi-criteria mathematical severity scoring engine
├── ml_model.py                # Random Forest & SVM classifiers with LOOCV
├── dl_model.py                # MobileNetV2 transfer learning & Grad-CAM visualizer
├── compare.py                 # Multi-paradigm comparative benchmark runner
├── severity_features.csv      # Extracted 9-feature matrix across dataset
├── comparison_table.csv       # Side-by-side inference results table
├── comparison_summary.txt     # Benchmark summary metrics & agreement rates
├── README.md                  # Comprehensive technical documentation
├── models/                    # Serialized models and confusion matrix plots
│   ├── rf_severity_model.joblib
│   ├── svm_severity_model.joblib
│   ├── mobilenetv2_severity.pth
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_svm.png
│   └── confusion_matrix_dl.png
└── gradcam/                   # Grad-CAM heatmap overlays
    ├── gradcam_healthy.jpg
    ├── gradcam_mild.jpg
    ├── gradcam_moderate.jpg
    └── gradcam_severe.jpg
```

---

## 6. Honest Limitations & Scientific Discussion

1. **Sample Size ($N=14$ Valid Hibiscus Leaves)**:
   - The dataset consists of 14 valid *Hibiscus rosa-sinensis* leaf photographs from `data/raw/` (after species check rejected non-hibiscus samples such as `mango.jpeg`).
   - For classical machine learning, Leave-One-Out Cross-Validation (LOOCV) was utilized to provide unbiased out-of-fold generalization estimates.
   - For deep learning, transfer learning from ImageNet-pretrained weights combined with extensive data augmentation (rotations, flips, color jitter) was applied; however, deep neural networks remain data-hungry, and a larger dataset ($\ge 100$ leaves) is strongly recommended for production deployment.

2. **Ground Truth Label Derivation**:
   - Because laboratory mass-spectrometry tissue nutrient assays were not conducted on this field dataset, ground-truth reference classes were established by visual symptom grading against botanical reference charts (cataloged in `data/labels.csv`) and calibrated multi-signal consensus.
   - Severity grades reflect macroscopic visual and architectural stress manifestations rather than exact elemental parts-per-million (ppm) ion counts.

3. **Class Distribution Skew**:
   - In the natural sample collection, the class distribution is naturally skewed toward intermediate deficiency states:
     - **Healthy**: 1 leaf
     - **Mild**: 7 leaves
     - **Moderate**: 5 leaves
     - **Severe**: 1 leaf
   - Class imbalance was addressed in training via inverse-frequency class weighting in the cross-entropy loss function.
   - In cross-validation, single-sample classes (Healthy and Severe) represent extreme edge cases that require additional photographic sampling to elevate statistical power.

4. **Paradigm Trade-offs**:
   - **Rule-Based Engine**: Completely transparent, deterministic, zero training overhead, mathematically explainable, and impossible to hallucinate. Highly recommended for botanical viva defense and field applications.
   - **Classical ML (Random Forest)**: Effectively captures non-linear thresholds between vein density and color indices (71.4% LOOCV accuracy), but requires clean extracted tabular features.
   - **Deep Learning (MobileNetV2)**: Operates directly on raw RGB pixels and discovers spatial visual patterns (demonstrated via Grad-CAM), but acts as a black box without Grad-CAM interpretation.
