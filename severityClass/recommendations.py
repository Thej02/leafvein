"""
severityClass/recommendations.py — Actionable botanical recommendations for Rosa-sinensis leaf health.

Maps each severity level ('Healthy', 'Mild', 'Moderate', 'Severe') to concise,
botanically sound diagnostic descriptions and corrective horticultural interventions.
"""

from typing import Dict, Any

SEVERITY_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "Healthy": {
        "status": "Optimal Plant Vigor",
        "botanical_summary": (
            "The leaf exhibits uniform chlorophyll distribution (high DGCI/ExG), "
            "normal vein architecture with healthy vascular density, and no significant chlorosis."
        ),
        "actions": [
            "Maintain current irrigation schedule; water deeply at soil level rather than over foliage.",
            "Ensure 6+ hours of direct sunlight daily for robust flowering.",
            "Apply balanced, slow-release fertilizer (e.g., NPK 10-10-10 or hibiscus-formulated 7-2-12) monthly during the active growing season.",
            "Continue periodic preventive scouting for common hibiscus pests (aphids, whiteflies, spider mites)."
        ],
        "nutrient_focus": "Balanced maintenance (NPK + trace minerals)",
        "urgency": "None (Routine maintenance)"
    },
    "Mild": {
        "status": "Early-Stage Stress / Incipient Chlorosis",
        "botanical_summary": (
            "Early warning signals detected: localized subtle yellowing or minor interveinal color divergence. "
            "Vascular vein architecture remains intact."
        ),
        "actions": [
            "Inspect soil drainage and test root-zone moisture; allow top 2 inches of soil to dry between waterings to prevent root suffocation.",
            "Test soil pH (ideal range for Rosa-sinensis is 6.0 - 6.8); improper pH locks out essential micronutrients even if present.",
            "Apply a dilute foliar micronutrient spray containing chelated iron (Fe) and magnesium sulfate (Epsom salts at 1 tbsp/gallon).",
            "Monitor affected leaves weekly to ensure symptoms do not progress down the branch."
        ],
        "nutrient_focus": "Magnesium (Mg) and Iron (Fe) foliar booster; check soil pH",
        "urgency": "Low to Moderate (Corrective care within 1-2 weeks)"
    },
    "Moderate": {
        "status": "Established Nutrient Deficiency / Vascular Contrast",
        "botanical_summary": (
            "Pronounced interveinal chlorosis (prominent green veins with pale/yellow intervening lamina) "
            "or substantial chlorophyll reduction across 25%-50% of the leaf surface."
        ),
        "actions": [
            "Apply chelated iron (Fe-EDDHA for alkaline soils, Fe-EDTA for neutral/acidic soils) as a soil drench around the drip line.",
            "Supplement with water-soluble magnesium sulfate (Epsom salt) to restore interveinal chloroplast activity.",
            "Feed with an elevated potassium hibiscus fertilizer (e.g. 10-5-20 or 9-3-13); potassium deficiency often accompanies magnesium stress.",
            "Check leaf undersides with a hand lens to rule out spider mite damage, which mimics interveinal stippling.",
            "Avoid heavy nitrogen overfeeding which triggers rapid growth vulnerable to micronutrient starvation."
        ],
        "nutrient_focus": "Chelated Iron (Fe), Magnesium (Mg), and Potassium (K)",
        "urgency": "Moderate to High (Remediate within 3-5 days)"
    },
    "Severe": {
        "status": "Advanced Chlorosis & Critical Vascular Degradation",
        "botanical_summary": (
            "Extensive chlorophyll breakdown (>50% yellowing), diminished vein density, "
            "and marked vascular contrast. Leaf tissue is at risk of necrosis and premature abscission."
        ),
        "actions": [
            "Perform soil leaching: flush the pot/soil thoroughly with clean water to remove excess salts or fertilizer buildup causing nutrient lock.",
            "Administer rapid-absorption foliar feed (comprehensive chelated micro-pack: Fe, Mg, Mn, Zn) in early morning or late evening.",
            "Apply liquid root-zone drench with kelp extract and chelated iron/magnesium.",
            "Prune necrotic or completely bleached foliage to reduce respiratory burden and redirect plant reserves to emerging shoots.",
            "Isolate plant if mottling or high spatial variance indicates viral vector or severe pest infestation.",
            "Conduct laboratory soil/tissue testing if recovery is not visible within 14 days."
        ],
        "nutrient_focus": "Emergency full-spectrum micronutrients (Fe, Mg, Mn, Zn) + root flush",
        "urgency": "Critical (Immediate intervention required)"
    }
}


def get_recommendation(severity_level: str) -> Dict[str, Any]:
    """
    Retrieve the actionable recommendation and botanical summary for a given severity level.

    Args:
        severity_level: One of 'Healthy', 'Mild', 'Moderate', 'Severe'.

    Returns:
        Dict containing status, botanical summary, actionable steps, nutrient focus, and urgency.
    """
    normalized = severity_level.strip().capitalize()
    if normalized not in SEVERITY_RECOMMENDATIONS:
        # Fallback if unmapped
        return {
            "status": f"Severity Level: {severity_level}",
            "botanical_summary": "Unclassified health condition.",
            "actions": ["Consult an agricultural extension specialist or conduct laboratory soil testing."],
            "nutrient_focus": "General diagnostic testing",
            "urgency": "Moderate"
        }
    return SEVERITY_RECOMMENDATIONS[normalized]
