"""
src/deficiency_typing.py — Deficiency-Type Identification (Stage 8.5)

Maps combinations of failed primary factors to one of a small set of named patterns,
using an explicit, non-ML rule table based on botanical literature for Hibiscus rosa-sinensis.

Outputs are always hedged as pattern matches, not diagnostic claims.
"""

from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.thresholds import (
    COLOR_SPATIAL_VARIANCE_MAX,
    INTERVEINAL_CONTRAST_THRESHOLD,
    YELLOW_RATIO_DEFICIENT,
)

def identify_deficiency(features: Dict[str, Any], failed_factors: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map failed factors to a known deficiency or damage pattern.
    Only to be invoked when Stage 8 returns NOT HEALTHY.

    Args:
        features: Feature dict from feature_extraction.
        failed_factors: List of flagged feature dicts from decision_engine.

    Returns:
        Dict containing:
            'pattern_name': Short name of the identified pattern (used for circle legend)
            'assessment_text': Full formatted assessment string for the text report
    """
    variance = features.get('color_spatial_variance', 0.0)
    contrast = features.get('interveinal_contrast', 0.0)
    yellow_ratio = features.get('yellow_pixel_ratio', 0.0)

    disclaimer = (
        "This is a pattern-matching estimate based on color and vein "
        "signals only — not a substitute for soil/tissue testing or expert diagnosis."
    )

    if variance > COLOR_SPATIAL_VARIANCE_MAX:
        pattern_name = "Possible pest/disease/damage"
        text = (
            f"Pattern consistent with {pattern_name.lower()}.\n"
            f"Color spatial variance ({variance:.1f}) is high (>{COLOR_SPATIAL_VARIANCE_MAX:.1f}), "
            f"indicating sharp, localized, patchy discoloration rather than diffuse nutrient deficiency.\n"
            f"Recommend closer visual inspection or expert consultation.\n"
            f"{disclaimer}"
        )
    elif contrast > INTERVEINAL_CONTRAST_THRESHOLD and yellow_ratio > YELLOW_RATIO_DEFICIENT:
        pattern_name = "Iron or magnesium deficiency pattern"
        text = (
            f"Pattern consistent with iron or magnesium deficiency (veins remaining green while surrounding tissue yellows).\n"
            f"Interveinal contrast ({contrast:.1f}) is high (>{INTERVEINAL_CONTRAST_THRESHOLD:.1f}) and "
            f"yellow pixel ratio ({yellow_ratio:.1%}) is elevated (>{YELLOW_RATIO_DEFICIENT:.1%}).\n"
            f"Leaf position on the plant (new growth vs. older growth) would help distinguish these — "
            f"iron deficiency typically appears on new growth, magnesium deficiency on older growth.\n"
            f"{disclaimer}"
        )
    elif yellow_ratio > YELLOW_RATIO_DEFICIENT and contrast <= INTERVEINAL_CONTRAST_THRESHOLD:
        pattern_name = "Nitrogen deficiency pattern"
        text = (
            f"Pattern consistent with nitrogen deficiency (diffuse, more uniform yellowing rather than sharply vein-bounded).\n"
            f"Yellow pixel ratio ({yellow_ratio:.1%}) is high (>{YELLOW_RATIO_DEFICIENT:.1%}) while "
            f"interveinal contrast ({contrast:.1f}) remains low-to-moderate (<={INTERVEINAL_CONTRAST_THRESHOLD:.1f}).\n"
            f"{disclaimer}"
        )
    else:
        pattern_name = "Unknown pattern"
        text = (
            f"Leaf flagged NOT HEALTHY but does not clearly match a known deficiency signature.\n"
            f"Recommend closer inspection.\n"
            f"{disclaimer}"
        )

    return {
        'pattern_name': pattern_name,
        'assessment_text': text
    }
