"""
src/decision_engine.py — Rule-based health verdict engine.

Takes a feature dict (from feature_extraction.py) and produces:
  1. A verdict: "Healthy", "Possibly Deficient", or "Deficient"
  2. A human-readable reasoning string explaining which features
     crossed which thresholds and what deficiency pattern that
     combination resembles.

This is a deterministic, explicit if-then rule engine.
There is no trained model, no fitted parameters, no ML of any kind.
All thresholds are read from config/thresholds.py and were calibrated
by transparent statistical methods documented in Stage 7.

Every verdict path produces a reasoning string — this is a product
requirement, not optional logging.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    VEIN_DENSITY_HEALTHY_LOW,
    VEIN_DENSITY_DEFICIENT,
    YELLOW_RATIO_POSSIBLY_DEFICIENT,
    YELLOW_RATIO_DEFICIENT,
    EXG_HEALTHY_LOW,
    DGCI_HEALTHY_LOW,
    MEAN_SATURATION_HEALTHY_LOW,
    VEIN_THICKNESS_HEALTHY_LOW,
    INTERVEINAL_CONTRAST_THRESHOLD,
)


# Verdict constants
VERDICT_HEALTHY = "Healthy"
VERDICT_POSSIBLY_DEFICIENT = "Possibly Deficient"
VERDICT_DEFICIENT = "Deficient"


def evaluate(features: dict) -> dict:
    """
    Evaluate leaf health based on extracted features using rule-based logic.

    The decision follows a multi-signal approach:
      1. Check individual features against thresholds
      2. Count how many signals indicate deficiency
      3. Determine verdict based on signal count and severity
      4. Generate a human-readable reasoning string

    Args:
        features: Dict of feature values from feature_extraction.extract_all_features().
                  Expected keys: 'vein_density', 'vein_thickness_avg', 'branch_point_count',
                  'mean_hue', 'mean_saturation', 'yellow_pixel_ratio',
                  'excess_green_index', 'dgci', 'interveinal_contrast'.

    Returns:
        Dict with keys:
          'verdict': str — one of VERDICT_HEALTHY, VERDICT_POSSIBLY_DEFICIENT, VERDICT_DEFICIENT
          'confidence_signals': int — how many individual features flagged deficiency
          'reasoning': str — human-readable explanation of the verdict
          'flags': list of dict — each flagged feature with its value, threshold, and meaning
    """
    flags = []

    # ── Vein density check ──────────────────────────────────────────────
    vein_density = features.get('vein_density', 0)

    if vein_density < VEIN_DENSITY_DEFICIENT:
        flags.append({
            'feature': 'vein_density',
            'value': vein_density,
            'threshold': VEIN_DENSITY_DEFICIENT,
            'direction': 'below',
            'severity': 'high',
            'meaning': (
                f"Vein density ({vein_density:.4f}) is well below the healthy baseline "
                f"({VEIN_DENSITY_DEFICIENT:.4f}). This suggests reduced vascular "
                f"architecture, which may indicate nutrient deficiency affecting "
                f"leaf vein development."
            ),
        })
    elif vein_density < VEIN_DENSITY_HEALTHY_LOW:
        flags.append({
            'feature': 'vein_density',
            'value': vein_density,
            'threshold': VEIN_DENSITY_HEALTHY_LOW,
            'direction': 'below',
            'severity': 'moderate',
            'meaning': (
                f"Vein density ({vein_density:.4f}) is below the healthy range "
                f"({VEIN_DENSITY_HEALTHY_LOW:.4f}). This is a mild reduction that "
                f"warrants monitoring."
            ),
        })

    # ── Yellow pixel ratio check ────────────────────────────────────────
    yellow_ratio = features.get('yellow_pixel_ratio', 0)

    if yellow_ratio > YELLOW_RATIO_DEFICIENT:
        flags.append({
            'feature': 'yellow_pixel_ratio',
            'value': yellow_ratio,
            'threshold': YELLOW_RATIO_DEFICIENT,
            'direction': 'above',
            'severity': 'high',
            'meaning': (
                f"Yellow pixel ratio ({yellow_ratio:.1%}) is significantly above "
                f"the healthy range (>{YELLOW_RATIO_DEFICIENT:.1%}). Extensive "
                f"yellowing (chlorosis) suggests a nutrient deficiency — pattern "
                f"consistent with nitrogen or magnesium deficiency."
            ),
        })
    elif yellow_ratio > YELLOW_RATIO_POSSIBLY_DEFICIENT:
        flags.append({
            'feature': 'yellow_pixel_ratio',
            'value': yellow_ratio,
            'threshold': YELLOW_RATIO_POSSIBLY_DEFICIENT,
            'direction': 'above',
            'severity': 'moderate',
            'meaning': (
                f"Yellow pixel ratio ({yellow_ratio:.1%}) is above the healthy "
                f"baseline (>{YELLOW_RATIO_POSSIBLY_DEFICIENT:.1%}). Some chlorosis "
                f"is present — early sign of possible nutrient deficiency."
            ),
        })

    # ── Excess Green Index check ────────────────────────────────────────
    exg = features.get('excess_green_index', 0)

    if exg < EXG_HEALTHY_LOW:
        flags.append({
            'feature': 'excess_green_index',
            'value': exg,
            'threshold': EXG_HEALTHY_LOW,
            'direction': 'below',
            'severity': 'moderate',
            'meaning': (
                f"Excess Green Index ({exg:.4f}) is below the healthy threshold "
                f"({EXG_HEALTHY_LOW:.4f}). This indicates reduced overall greenness, "
                f"consistent with chlorophyll loss from nutrient stress."
            ),
        })

    # ── DGCI check ─────────────────────────────────────────────────────
    dgci = features.get('dgci', 0)

    if dgci < DGCI_HEALTHY_LOW:
        flags.append({
            'feature': 'dgci',
            'value': dgci,
            'threshold': DGCI_HEALTHY_LOW,
            'direction': 'below',
            'severity': 'moderate',
            'meaning': (
                f"Dark Green Color Index ({dgci:.4f}) is below the healthy "
                f"threshold ({DGCI_HEALTHY_LOW:.4f}). DGCI correlates with leaf "
                f"nitrogen content — low values suggest a pale, chlorotic leaf."
            ),
        })

    # ── Mean saturation check ──────────────────────────────────────────
    mean_sat = features.get('mean_saturation', 0)

    if mean_sat < MEAN_SATURATION_HEALTHY_LOW:
        flags.append({
            'feature': 'mean_saturation',
            'value': mean_sat,
            'threshold': MEAN_SATURATION_HEALTHY_LOW,
            'direction': 'below',
            'severity': 'moderate',
            'meaning': (
                f"Mean saturation ({mean_sat:.1f}) is below the healthy range "
                f"({MEAN_SATURATION_HEALTHY_LOW:.1f}). Low color saturation "
                f"indicates washed-out coloring, consistent with chlorosis."
            ),
        })

    # ── Vein thickness check ───────────────────────────────────────────
    vein_thickness = features.get('vein_thickness_avg', 0)

    if vein_thickness < VEIN_THICKNESS_HEALTHY_LOW and vein_thickness > 0:
        flags.append({
            'feature': 'vein_thickness_avg',
            'value': vein_thickness,
            'threshold': VEIN_THICKNESS_HEALTHY_LOW,
            'direction': 'below',
            'severity': 'low',
            'meaning': (
                f"Average vein thickness ({vein_thickness:.2f}px) is below the "
                f"healthy baseline ({VEIN_THICKNESS_HEALTHY_LOW:.2f}px). Thinner "
                f"veins may indicate reduced vascular development."
            ),
        })

    # ── Interveinal contrast check ─────────────────────────────────────
    interveinal = features.get('interveinal_contrast', 0)

    if interveinal > INTERVEINAL_CONTRAST_THRESHOLD:
        flags.append({
            'feature': 'interveinal_contrast',
            'value': interveinal,
            'threshold': INTERVEINAL_CONTRAST_THRESHOLD,
            'direction': 'above',
            'severity': 'moderate',
            'meaning': (
                f"Interveinal contrast ({interveinal:.1f}) exceeds the threshold "
                f"({INTERVEINAL_CONTRAST_THRESHOLD:.1f}). This 'green veins with "
                f"yellowing between veins' pattern is a classic signature of "
                f"magnesium or iron deficiency."
            ),
        })

    # ── Determine verdict ──────────────────────────────────────────────
    high_severity_count = sum(1 for f in flags if f['severity'] == 'high')
    moderate_severity_count = sum(1 for f in flags if f['severity'] == 'moderate')
    total_flags = len(flags)

    if high_severity_count >= 2 or (high_severity_count >= 1 and moderate_severity_count >= 2):
        verdict = VERDICT_DEFICIENT
    elif high_severity_count >= 1 or moderate_severity_count >= 2 or total_flags >= 3:
        verdict = VERDICT_POSSIBLY_DEFICIENT
    elif total_flags >= 1:
        verdict = VERDICT_POSSIBLY_DEFICIENT
    else:
        verdict = VERDICT_HEALTHY

    # ── Build reasoning string ─────────────────────────────────────────
    reasoning = _build_reasoning(verdict, flags, features)

    return {
        'verdict': verdict,
        'confidence_signals': total_flags,
        'reasoning': reasoning,
        'flags': flags,
    }


def _build_reasoning(verdict: str, flags: list, features: dict) -> str:
    """
    Build a human-readable reasoning string explaining the verdict.

    Uses hedging language ("pattern consistent with…", "may indicate…")
    to avoid making diagnostic claims — per goal.md constraint #4.

    Args:
        verdict: The determined verdict string.
        flags: List of flagged feature dicts.
        features: Original feature dict (for reporting values even if not flagged).

    Returns:
        Multi-line human-readable reasoning string.
    """
    lines = []

    lines.append(f"VERDICT: {verdict}")
    lines.append("")

    if verdict == VERDICT_HEALTHY:
        lines.append(
            "All measured features are within the expected healthy range for "
            "Rosa-sinensis. No signs of nutrient deficiency detected."
        )
        lines.append("")
        lines.append("Feature summary:")
        lines.append(f"  • Vein density: {features.get('vein_density', 0):.4f}")
        lines.append(f"  • Yellow pixel ratio: {features.get('yellow_pixel_ratio', 0):.1%}")
        lines.append(f"  • Excess Green Index: {features.get('excess_green_index', 0):.4f}")
        lines.append(f"  • DGCI: {features.get('dgci', 0):.4f}")
        lines.append(f"  • Mean saturation: {features.get('mean_saturation', 0):.1f}")
    else:
        lines.append(
            f"{len(flags)} feature(s) outside the healthy baseline range:"
        )
        lines.append("")

        for i, flag in enumerate(flags, 1):
            lines.append(f"  {i}. {flag['meaning']}")
            lines.append("")

        # Identify potential deficiency patterns
        patterns = _identify_deficiency_patterns(flags)
        if patterns:
            lines.append("Possible deficiency patterns:")
            for pattern in patterns:
                lines.append(f"  -> {pattern}")
            lines.append("")

        lines.append(
            "NOTE: This assessment is based on image analysis of leaf color "
            "and vein architecture. It identifies patterns consistent with "
            "known nutrient deficiency symptoms but is not a definitive "
            "diagnosis. For confirmation, consult a plant specialist or "
            "conduct soil/tissue testing."
        )

    return "\n".join(lines)


def _identify_deficiency_patterns(flags: list) -> list:
    """
    Match the combination of flagged features to known deficiency patterns.

    This uses the Reference Vocabulary from goal.md to describe patterns
    in appropriate botanical terms with hedging language.

    Args:
        flags: List of flagged feature dicts.

    Returns:
        List of pattern description strings.
    """
    patterns = []
    flag_names = {f['feature'] for f in flags}

    # Nitrogen deficiency: uniform yellowing, low ExG, low DGCI, reduced veins
    if ('yellow_pixel_ratio' in flag_names and
            ('excess_green_index' in flag_names or 'dgci' in flag_names)):
        if 'interveinal_contrast' not in flag_names:
            patterns.append(
                "Pattern consistent with nitrogen deficiency: uniform chlorosis "
                "(general yellowing without preferential vein retention) combined "
                "with reduced green color indices."
            )

    # Magnesium deficiency: interveinal chlorosis (veins stay green, tissue yellows)
    if 'interveinal_contrast' in flag_names and 'yellow_pixel_ratio' in flag_names:
        patterns.append(
            "Pattern consistent with magnesium deficiency: interveinal chlorosis "
            "(tissue between veins is yellowing while veins retain green color), "
            "a classic Mg-deficiency signature in older leaves."
        )

    # Iron deficiency: similar interveinal pattern but typically on younger leaves
    if 'interveinal_contrast' in flag_names and 'mean_saturation' in flag_names:
        patterns.append(
            "Pattern may also be consistent with iron deficiency: interveinal "
            "chlorosis with reduced color saturation. Iron deficiency typically "
            "appears on younger leaves first (not distinguishable from Mg without "
            "knowing leaf age)."
        )

    # Vein architecture degradation
    if 'vein_density' in flag_names:
        patterns.append(
            "Reduced vein density may indicate impaired vascular development, "
            "which can result from prolonged nutrient stress or other "
            "environmental factors."
        )

    if not patterns:
        patterns.append(
            "The observed feature anomalies do not clearly match a single "
            "known deficiency pattern. Multiple mild deviations detected — "
            "continued monitoring recommended."
        )

    return patterns
