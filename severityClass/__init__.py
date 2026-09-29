"""
severityClass package — Rosa-sinensis leaf health severity classification.

Provides:
- rule_based: Rule-based severity scoring and multi-factor thresholding.
- ml_model: Classical machine learning (Random Forest & SVM) severity classifiers.
- dl_model: Deep learning (MobileNetV2) transfer learning and Grad-CAM interpretability.
- recommendations: Actionable botanical recommendations per severity grade.
- compare: Side-by-side comparison across all three methodologies.
"""

from severityClass.recommendations import get_recommendation, SEVERITY_RECOMMENDATIONS

__all__ = [
    'get_recommendation',
    'SEVERITY_RECOMMENDATIONS',
]
