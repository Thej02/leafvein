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
    YELLOW_RATIO_DEFICIENT,
    EXG_HEALTHY_LOW,
    DGCI_HEALTHY_LOW,
    INTERVEINAL_CONTRAST_THRESHOLD,
    COLOR_SPATIAL_VARIANCE_MAX,
    VEIN_DENSITY_DEFICIENT,
    VEIN_THICKNESS_DEFICIENT_HIGH,
)


# Verdict constants
VERDICT_HEALTHY = "Leaf Health Status: HEALTHY"
VERDICT_NOT_HEALTHY = "Leaf Health Status: NOT HEALTHY"


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
          'verdict': str — one of VERDICT_HEALTHY, VERDICT_NOT_HEALTHY
          'confidence_signals': int — how many individual features flagged deficiency
          'reasoning': str — human-readable explanation of the verdict
          'flags': list of dict — each flagged feature with its value, threshold, and meaning
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Primary Factors (Color / Chlorosis)
    # ─────────────────────────────────────────────────────────────────────────
    primary_flags = []

    yellow_ratio = features.get('yellow_pixel_ratio', 0.0)
    if yellow_ratio > YELLOW_RATIO_DEFICIENT:
        primary_flags.append({
            'feature': 'yellow_pixel_ratio',
            'value': yellow_ratio,
            'threshold': YELLOW_RATIO_DEFICIENT,
            'type': 'primary',
            'meaning': (
                f"Yellow pixel ratio ({yellow_ratio:.1%}) is above the healthy "
                f"baseline (>{YELLOW_RATIO_DEFICIENT:.1%}). Extensive yellowing "
                f"(chlorosis) suggests a nutrient deficiency."
            ),
        })

    exg = features.get('excess_green_index', 1.0)
    if exg < EXG_HEALTHY_LOW:
        primary_flags.append({
            'feature': 'excess_green_index',
            'value': exg,
            'threshold': EXG_HEALTHY_LOW,
            'type': 'primary',
            'meaning': (
                f"Excess Green Index ({exg:.4f}) is below the healthy "
                f"threshold ({EXG_HEALTHY_LOW:.4f}). This indicates reduced "
                f"overall greenness, consistent with chlorophyll loss from nutrient stress."
            ),
        })

    dgci = features.get('dgci', 1.0)
    if dgci < DGCI_HEALTHY_LOW:
        primary_flags.append({
            'feature': 'dgci',
            'value': dgci,
            'threshold': DGCI_HEALTHY_LOW,
            'type': 'primary',
            'meaning': (
                f"Dark Green Color Index ({dgci:.4f}) is below the healthy "
                f"threshold ({DGCI_HEALTHY_LOW:.4f}). DGCI correlates with overall "
                f"leaf health — low values suggest a pale, chlorotic leaf."
            ),
        })

    interveinal = features.get('interveinal_contrast', 0.0)
    if interveinal > INTERVEINAL_CONTRAST_THRESHOLD:
        primary_flags.append({
            'feature': 'interveinal_contrast',
            'value': interveinal,
            'threshold': INTERVEINAL_CONTRAST_THRESHOLD,
            'type': 'primary',
            'meaning': (
                f"Interveinal contrast ({interveinal:.1f}) exceeds the threshold "
                f"({INTERVEINAL_CONTRAST_THRESHOLD:.1f}). This 'green veins with "
                f"yellowing between veins' pattern is a signature of nutrient stress."
            ),
        })
    variance = features.get('color_spatial_variance', 0.0)
    if variance > COLOR_SPATIAL_VARIANCE_MAX:
        primary_flags.append({
            'feature': 'color_spatial_variance',
            'value': variance,
            'threshold': COLOR_SPATIAL_VARIANCE_MAX,
            'type': 'primary',
            'meaning': (
                f"Color spatial variance ({variance:.1f}) is very high "
                f"(>{COLOR_SPATIAL_VARIANCE_MAX:.1f}). This suggests sharp, localized "
                f"patches (e.g., pest damage, fungal spotting, or physical injury) "
                f"rather than diffuse systemic nutrient deficiency. Closer inspection recommended."
            ),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Secondary Factors (Vein Geometry)
    # ─────────────────────────────────────────────────────────────────────────
    secondary_flags = []

    vein_density = features.get('vein_density', 1.0)
    if vein_density < VEIN_DENSITY_DEFICIENT:
        secondary_flags.append({
            'feature': 'vein_density',
            'value': vein_density,
            'threshold': VEIN_DENSITY_DEFICIENT,
            'type': 'secondary',
            'meaning': (
                f"Vein density ({vein_density:.4f}) is below the healthy baseline "
                f"({VEIN_DENSITY_DEFICIENT:.4f}). This may indicate impaired vascular "
                f"development."
            ),
        })

    vein_thickness = features.get('vein_thickness_avg', 0.0)
    if vein_thickness > VEIN_THICKNESS_DEFICIENT_HIGH:
        secondary_flags.append({
            'feature': 'vein_thickness_avg',
            'value': vein_thickness,
            'threshold': VEIN_THICKNESS_DEFICIENT_HIGH,
            'type': 'secondary',
            'meaning': (
                f"Average vein thickness ({vein_thickness:.2f}px) is above the healthy baseline "
                f"({VEIN_THICKNESS_DEFICIENT_HIGH:.2f}px). Enlarged, corky veins can be a "
                f"secondary sign of specific nutrient stresses."
            ),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Combine and evaluate
    # ─────────────────────────────────────────────────────────────────────────
    if len(primary_flags) > 0:
        verdict = VERDICT_NOT_HEALTHY
        flags = primary_flags + secondary_flags
    else:
        verdict = VERDICT_HEALTHY
        flags = []  # Secondary flags alone do not trigger failure
        
    total_flags = len(flags)

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
        lines.append(f"  • Color Spatial Variance: {features.get('color_spatial_variance', 0):.1f}")
    else:
        lines.append(
            f"{len(flags)} feature(s) outside the healthy baseline range:"
        )
        lines.append("")

        for i, flag in enumerate(flags, 1):
            lines.append(f"  {i}. {flag['meaning']}")
            lines.append("")

        lines.append(
            "Deficiency-type analysis: not yet implemented — coming in a later stage."
        )

    return "\n".join(lines)
