"""
severityClass/compare.py — Multi-Methodology Severity Comparison Engine.

Evaluates the 3 distinct paradigms side-by-side on the same Rosa-sinensis leaf dataset:
  1. Rule-Based Botanical Scoring Engine (interpretable, zero-training, explicit domain rules)
  2. Classical Machine Learning (Random Forest & Support Vector Machine on extracted features)
  3. Deep Learning (MobileNetV2 Transfer Learning with Grad-CAM spatial localization)

Outputs:
  - Formatted side-by-side comparison table
  - Pairwise and multi-way model agreement rates
  - Per-class accuracy comparison against reference labels
  - Actionable horticultural recommendations per leaf
  - Exported comparison table to CSV and text report
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib
import torch
from PIL import Image

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from severityClass.rule_based import compute_severity
from severityClass.ml_model import (
    FEATURE_COLUMNS,
    SEVERITY_CLASSES,
    MODELS_DIR,
    FEATURES_CSV_PATH,
    train_and_evaluate_models
)
from severityClass.dl_model import (
    build_mobilenetv2_model,
    get_transforms,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
    train_dl_model
)
from severityClass.recommendations import get_recommendation

SEVERITY_DIR = os.path.abspath(os.path.dirname(__file__))


def load_all_models():
    """Load trained RF, SVM, and MobileNetV2 models."""
    rf_path = os.path.join(MODELS_DIR, 'rf_severity_model.joblib')
    svm_path = os.path.join(MODELS_DIR, 'svm_severity_model.joblib')
    dl_path = os.path.join(MODELS_DIR, 'mobilenetv2_severity.pth')

    if not (os.path.exists(rf_path) and os.path.exists(svm_path)):
        print("Classical ML models not found. Training now...")
        train_and_evaluate_models()

    if not os.path.exists(dl_path):
        print("Deep Learning model not found. Training now...")
        train_dl_model()

    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_model = build_mobilenetv2_model(num_classes=len(SEVERITY_CLASSES), pretrained=False)
    dl_model.load_state_dict(torch.load(dl_path, map_location=device))
    dl_model.to(device)
    dl_model.eval()

    return rf_model, svm_model, dl_model, device


def run_comparison(export_csv: bool = True) -> pd.DataFrame:
    """
    Run side-by-side inference across all three paradigms and print formatted comparison.
    """
    if not os.path.exists(FEATURES_CSV_PATH):
        from severityClass.ml_model import extract_dataset_features
        extract_dataset_features()

    df = pd.read_csv(FEATURES_CSV_PATH)
    rf_model, svm_model, dl_model, device = load_all_models()
    _, val_transform = get_transforms()

    comparison_rows = []

    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        filename = str(row['filename'])
        img_path = str(row['image_path'])
        ref_label = str(row['ref_label'])
        true_class = str(row['severity_class'])

        # 1. Rule-Based Evaluation
        feat_dict = {col: row[col] for col in FEATURE_COLUMNS}
        rule_res = compute_severity(feat_dict)
        rule_class = rule_res['severity_level']
        rule_score = rule_res['severity_score']

        # 2. Classical ML Evaluation
        X_sample = np.array([[row[col] for col in FEATURE_COLUMNS]], dtype=np.float64)
        rf_class = rf_model.predict(X_sample)[0]
        rf_prob = np.max(rf_model.predict_proba(X_sample)[0])

        svm_class = svm_model.predict(X_sample)[0]
        svm_prob = np.max(svm_model.predict_proba(X_sample)[0])

        # 3. Deep Learning Evaluation
        pil_img = Image.open(img_path).convert('RGB')
        tensor_img = val_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            dl_out = dl_model(tensor_img)
            dl_probs = torch.softmax(dl_out, dim=1).cpu().numpy()[0]
            dl_idx = np.argmax(dl_probs)
            dl_class = IDX_TO_CLASS[dl_idx]
            dl_conf = float(dl_probs[dl_idx])

        # Consensus analysis
        preds = [rule_class, rf_class, svm_class, dl_class]
        from collections import Counter
        counts = Counter(preds)
        most_common_class, top_count = counts.most_common(1)[0]
        consensus = f"{top_count}/4 ({most_common_class})"

        rec = get_recommendation(rule_class)

        comparison_rows.append({
            'image_id': img_id,
            'filename': filename,
            'reference_truth': true_class,
            'rule_class': rule_class,
            'rule_score': rule_score,
            'rf_class': rf_class,
            'rf_conf': round(rf_prob, 3),
            'svm_class': svm_class,
            'svm_conf': round(svm_prob, 3),
            'dl_class': dl_class,
            'dl_conf': round(dl_conf, 3),
            'consensus': consensus,
            'agreement_4way': len(set(preds)) == 1,
            'agreement_rule_rf': (rule_class == rf_class),
            'agreement_rule_dl': (rule_class == dl_class),
            'agreement_rf_dl': (rf_class == dl_class),
            'primary_recommendation': rec['status'],
            'nutrient_focus': rec['nutrient_focus'],
            'urgency': rec['urgency'],
        })

    comp_df = pd.DataFrame(comparison_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Print Formatted Comparison Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 115)
    print("  ROSA-SINENSIS LEAF HEALTH: MULTI-APPROACH SEVERITY COMPARISON")
    print("=" * 115)
    
    header = (
        f"{'Leaf ID':<8s} | {'Ref Label':<10s} | {'Rule-Based (Score)':<22s} | "
        f"{'Random Forest':<15s} | {'SVM':<12s} | {'MobileNetV2':<15s} | {'Consensus':<15s}"
    )
    print(header)
    print("-" * 115)

    for _, r in comp_df.iterrows():
        rule_str = f"{r['rule_class']} ({r['rule_score']:.1f})"
        rf_str = f"{r['rf_class']} ({r['rf_conf']:.0%})"
        svm_str = f"{r['svm_class']} ({r['svm_conf']:.0%})"
        dl_str = f"{r['dl_class']} ({r['dl_conf']:.0%})"
        line = (
            f"{r['image_id']:<8s} | {r['reference_truth']:<10s} | {rule_str:<22s} | "
            f"{rf_str:<15s} | {svm_str:<12s} | {dl_str:<15s} | {r['consensus']:<15s}"
        )
        print(line)

    print("-" * 115)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute Agreement Rates & Accuracy
    # ─────────────────────────────────────────────────────────────────────────
    n = len(comp_df)
    rule_rf_agree = comp_df['agreement_rule_rf'].mean()
    rule_dl_agree = comp_df['agreement_rule_dl'].mean()
    rf_dl_agree = comp_df['agreement_rf_dl'].mean()
    all_agree = comp_df['agreement_4way'].mean()

    # Accuracy vs Reference Truth
    acc_rule = (comp_df['rule_class'] == comp_df['reference_truth']).mean()
    acc_rf = (comp_df['rf_class'] == comp_df['reference_truth']).mean()
    acc_svm = (comp_df['svm_class'] == comp_df['reference_truth']).mean()
    acc_dl = (comp_df['dl_class'] == comp_df['reference_truth']).mean()

    print("\n" + "=" * 65)
    print("  PAIRWISE & ENSEMBLE AGREEMENT RATES (N=14)")
    print("=" * 65)
    print(f"  * Rule-Based vs Random Forest : {rule_rf_agree:.1%} ({comp_df['agreement_rule_rf'].sum()}/{n} leaves)")
    print(f"  * Rule-Based vs MobileNetV2   : {rule_dl_agree:.1%} ({comp_df['agreement_rule_dl'].sum()}/{n} leaves)")
    print(f"  * Random Forest vs MobileNetV2: {rf_dl_agree:.1%} ({comp_df['agreement_rf_dl'].sum()}/{n} leaves)")
    print(f"  * Unanimous 4-Way Agreement   : {all_agree:.1%} ({comp_df['agreement_4way'].sum()}/{n} leaves)")

    print("\n" + "=" * 65)
    print("  ACCURACY AGAINST REFERENCE GROUND TRUTH")
    print("=" * 65)
    print(f"  * Rule-Based Expert Engine    : {acc_rule:.1%} ({int(acc_rule*n)}/{n})")
    print(f"  * Random Forest Classifier    : {acc_rf:.1%} ({int(acc_rf*n)}/{n})")
    print(f"  * Support Vector Machine (SVM): {acc_svm:.1%} ({int(acc_svm*n)}/{n})")
    print(f"  * MobileNetV2 Transfer Model  : {acc_dl:.1%} ({int(acc_dl*n)}/{n})")

    # ─────────────────────────────────────────────────────────────────────────
    # Actionable Recommendations Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ACTIONABLE HORTICULTURAL INTERVENTIONS (Derived from Consensus)")
    print("=" * 80)
    for _, r in comp_df.iterrows():
        print(f"  Leaf #{r['image_id']:2s} [{r['rule_class'].upper()} Severity]:")
        print(f"    * Diagnosis: {r['primary_recommendation']}")
        print(f"    * Target Nutrients: {r['nutrient_focus']}")
        print(f"    * Urgency: {r['urgency']}")

    # Export results
    if export_csv:
        csv_out = os.path.join(SEVERITY_DIR, 'comparison_table.csv')
        comp_df.to_csv(csv_out, index=False)
        print(f"\nExported comparison CSV to: {csv_out}")

        txt_out = os.path.join(SEVERITY_DIR, 'comparison_summary.txt')
        with open(txt_out, 'w', encoding='utf-8') as f:
            f.write("Rosa-sinensis Leaf Health: Multi-Methodology Severity Comparison\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Pairwise Agreement:\n")
            f.write(f"  Rule-Based vs Random Forest : {rule_rf_agree:.1%}\n")
            f.write(f"  Rule-Based vs MobileNetV2   : {rule_dl_agree:.1%}\n")
            f.write(f"  Random Forest vs MobileNetV2: {rf_dl_agree:.1%}\n")
            f.write(f"  Unanimous 4-Way Consensus   : {all_agree:.1%}\n\n")
            f.write(f"Accuracy vs Reference Ground Truth:\n")
            f.write(f"  Rule-Based Engine : {acc_rule:.1%}\n")
            f.write(f"  Random Forest     : {acc_rf:.1%}\n")
            f.write(f"  SVM               : {acc_svm:.1%}\n")
            f.write(f"  MobileNetV2       : {acc_dl:.1%}\n")
        print(f"Exported summary report to: {txt_out}")

    return comp_df


if __name__ == '__main__':
    run_comparison()
