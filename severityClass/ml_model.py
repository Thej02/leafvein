"""
severityClass/ml_model.py — Classical Machine Learning Severity Classifiers (Random Forest & SVM).

Extracts botanical features across all available Rosa-sinensis leaf images,
saves the dataset to CSV, trains Random Forest and SVM classifiers,
and reports out-of-fold cross-validation metrics (Precision, Recall, F1, Confusion Matrix).
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocessing import (
    load_image,
    resize_to_working_resolution,
    denoise,
    normalize_brightness,
)
from src.segmentation import segment_leaf
from src.species_check import verify_hibiscus_species
from src.vein_extraction import extract_veins
from src.feature_extraction import extract_all_features
from severityClass.rule_based import compute_severity
from severityClass.recommendations import get_recommendation

# Target directory for models and data
SEVERITY_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(SEVERITY_DIR, 'models')
FEATURES_CSV_PATH = os.path.join(SEVERITY_DIR, 'severity_features.csv')

SEVERITY_CLASSES = ['Healthy', 'Mild', 'Moderate', 'Severe']

FEATURE_COLUMNS = [
    'yellow_pixel_ratio',
    'interveinal_contrast',
    'vein_density',
    'vein_thickness_avg',
    'dgci',
    'excess_green_index',
    'mean_hue',
    'mean_saturation',
    'color_spatial_variance'
]


def extract_dataset_features(image_dir: str = None) -> pd.DataFrame:
    """
    Extract leaf features and assign severity labels for all raw images.

    Args:
        image_dir: Directory containing raw .jpeg/.jpg leaf images. Defaults to 'data/raw'.

    Returns:
        pd.DataFrame containing feature columns, severity scores, and severity classes.
    """
    if image_dir is None:
        image_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpeg')) + glob.glob(os.path.join(image_dir, '*.jpg')))
    
    # Load labels.csv if present for ground-truth reference
    labels_csv_path = os.path.join(PROJECT_ROOT, 'data', 'labels.csv')
    ground_truth_map = {}
    if os.path.exists(labels_csv_path):
        ref_df = pd.read_csv(labels_csv_path)
        for _, row in ref_df.iterrows():
            img_id = str(row['image_id']).strip()
            ground_truth_map[img_id] = str(row['label']).strip().lower()

    rows = []
    print(f"Extracting features from {len(image_paths)} image(s) in {image_dir}...")

    for path in image_paths:
        filename = os.path.basename(path)
        img_id = os.path.splitext(filename)[0]

        # Skip known non-hibiscus test confounders like mango.jpeg
        if 'mango' in img_id.lower():
            print(f"  [Skip] Non-hibiscus test confounder: {filename}")
            continue

        try:
            raw = load_image(path)
            resized = resize_to_working_resolution(raw)
            denoised = denoise(resized)
            preprocessed = normalize_brightness(denoised)

            # Segment & Species check
            seg = segment_leaf(preprocessed)
            mask = seg['mask']
            species = verify_hibiscus_species(mask)
            if not species['is_hibiscus']:
                print(f"  [Reject] {filename} failed hibiscus species verification.")
                continue

            leaf_area = cv2.countNonZero(mask)
            vein_res = extract_veins(preprocessed, mask)
            feats = extract_all_features(preprocessed, mask, vein_res, leaf_area)

            # Rule-based severity evaluation
            sev = compute_severity(feats)
            score = sev['severity_score']
            rule_level = sev['severity_level']

            # Determine aligned ground-truth severity class
            # Map legacy 3-class ground-truth if available:
            #   - 'healthy' -> 'Healthy'
            #   - 'possibly_deficient' -> 'Mild' or 'Moderate' based on severity score
            #   - 'deficient' -> 'Moderate' or 'Severe' based on severity score
            ref_label = ground_truth_map.get(img_id, None)
            if ref_label == 'healthy':
                assigned_class = 'Healthy' if score < 25.0 else 'Mild'
            elif ref_label == 'possibly_deficient':
                assigned_class = 'Mild' if score < 45.0 else 'Moderate'
            elif ref_label == 'deficient':
                assigned_class = 'Severe' if score >= 70.0 else 'Moderate'
            else:
                # If uncataloged in labels.csv (e.g. images 9-16), use the calibrated rule-based consensus
                assigned_class = rule_level

            row = {
                'image_id': img_id,
                'filename': filename,
                'image_path': path,
                'severity_score': score,
                'severity_class': assigned_class,
                'rule_class': rule_level,
                'ref_label': ref_label or 'unlabeled',
            }
            for col in FEATURE_COLUMNS:
                row[col] = feats.get(col, 0.0)

            rows.append(row)
            print(f"  -> {filename:10s} | Area: {leaf_area:6d}px | Score: {score:5.1f} | Class: {assigned_class}")

        except Exception as e:
            print(f"  [Error] Failed processing {filename}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(SEVERITY_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"\nSaved {len(df)} feature records to {FEATURES_CSV_PATH}")
    return df


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: str):
    """Save a clean visual confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True Severity Class',
        xlabel='Predicted Severity Class'
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def train_and_evaluate_models(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Train Random Forest and SVM classifiers using cross-validation.
    Reports per-class precision, recall, F1, and confusion matrices.
    """
    if df is None:
        if not os.path.exists(FEATURES_CSV_PATH):
            df = extract_dataset_features()
        else:
            df = pd.read_csv(FEATURES_CSV_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Classical Machine Learning Severity Classification")
    print("=" * 70)

    # Class distribution check
    class_counts = df['severity_class'].value_counts()
    print("\nDataset Class Distribution:")
    for cls in SEVERITY_CLASSES:
        print(f"  - {cls:10s}: {class_counts.get(cls, 0):2d} sample(s)")

    # Prepare features and labels as pure numpy arrays (avoid pyarrow indexing incompatibilities)
    X = np.asarray(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    y = np.asarray(df['severity_class'].to_numpy(dtype=str))

    # Determine unique classes present in dataset
    present_classes = [c for c in SEVERITY_CLASSES if c in np.unique(y)]

    print(f"\nPerforming Leave-One-Out Cross-Validation (LOOCV) on all {len(df)} samples...")
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()

    # 1. Random Forest Classifier
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2, random_state=42))
    ])

    # 2. Support Vector Machine (RBF kernel)
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42))
    ])

    # Out-of-fold LOOCV predictions
    y_pred_rf = np.empty_like(y)
    y_pred_svm = np.empty_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_pipeline.fit(X_train, y_train)
        svm_pipeline.fit(X_train, y_train)

        y_pred_rf[test_idx] = rf_pipeline.predict(X_test)
        y_pred_svm[test_idx] = svm_pipeline.predict(X_test)

    # Fit final models on full dataset for export
    rf_pipeline.fit(X, y)
    svm_pipeline.fit(X, y)

    rf_model_path = os.path.join(MODELS_DIR, 'rf_severity_model.joblib')
    svm_model_path = os.path.join(MODELS_DIR, 'svm_severity_model.joblib')
    joblib.dump(rf_pipeline, rf_model_path)
    joblib.dump(svm_pipeline, svm_model_path)
    print(f"Saved trained models:\n  -> {rf_model_path}\n  -> {svm_model_path}")

    # Metrics computation
    results = {}
    models = [
        ('Random Forest', y_pred_rf, 'confusion_matrix_rf.png'),
        ('Support Vector Machine (SVM)', y_pred_svm, 'confusion_matrix_svm.png'),
    ]

    for model_name, y_pred, cm_filename in models:
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, labels=present_classes, output_dict=True, zero_division=0)
        report_text = classification_report(y, y_pred, labels=present_classes, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=present_classes)

        cm_save_path = os.path.join(MODELS_DIR, cm_filename)
        plot_confusion_matrix(cm, present_classes, f"{model_name} Confusion Matrix", cm_save_path)

        print("\n" + "-" * 70)
        print(f"Model: {model_name}")
        print(f"Cross-Validation Accuracy: {acc:.1%}")
        print("-" * 70)
        print("Classification Report:")
        print(report_text)
        print("Confusion Matrix:")
        # Pretty print text confusion matrix
        header = f"{'':12s}" + "".join([f"{c:>10s}" for c in present_classes])
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{present_classes[i]:12s}" + "".join([f"{val:>10d}" for val in row])
            print(row_str)
        print(f"Saved plot: {cm_save_path}")

        results[model_name] = {
            'accuracy': acc,
            'report': report,
            'report_text': report_text,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'cm_plot_path': cm_save_path,
        }

    return results


def predict_leaf(image_path: str, model_type: str = 'rf') -> Dict[str, Any]:
    """
    Predict severity class of an individual leaf image using a trained ML model.

    Args:
        image_path: Path to the leaf image.
        model_type: 'rf' for Random Forest or 'svm' for SVM.

    Returns:
        Dict with predicted severity class, probabilities, and actionable recommendation.
    """
    model_file = 'rf_severity_model.joblib' if model_type.lower() == 'rf' else 'svm_severity_model.joblib'
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        train_and_evaluate_models()

    pipeline = joblib.load(model_path)

    # Extract features using existing pipeline
    raw = load_image(image_path)
    resized = resize_to_working_resolution(raw)
    denoised = denoise(resized)
    prep = normalize_brightness(denoised)

    seg = segment_leaf(prep)
    mask = seg['mask']
    leaf_area = cv2.countNonZero(mask)
    vein_res = extract_veins(prep, mask)
    feats = extract_all_features(prep, mask, vein_res, leaf_area)

    feature_vector = np.array([[feats.get(c, 0.0) for c in FEATURE_COLUMNS]])
    pred_class = pipeline.predict(feature_vector)[0]
    probs = pipeline.predict_proba(feature_vector)[0]
    classes = pipeline.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
    rec = get_recommendation(pred_class)

    return {
        'image_path': image_path,
        'model_used': model_type.upper(),
        'predicted_class': pred_class,
        'probabilities': prob_dict,
        'recommendation': rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Severity Classifier (Random Forest & SVM)")
    parser.add_argument('--train', action='store_true', help="Extract features and train models")
    parser.add_argument('--predict', type=str, default=None, help="Predict severity for a single leaf image")
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'svm'], help="Model type to use for prediction")
    args = parser.parse_args()

    if args.predict:
        res = predict_leaf(args.predict, args.model)
        print("=" * 70)
        print(f"ML Severity Prediction ({res['model_used']})")
        print("=" * 70)
        print(f"Image:           {args.predict}")
        print(f"Predicted Class: {res['predicted_class']}")
        print("Class Probabilities:")
        for k, v in res['probabilities'].items():
            print(f"  - {k:10s}: {v:.1%}")
        rec = res['recommendation']
        print("-" * 70)
        print(f"Recommendation: {rec['status']}")
        print(f"Focus:          {rec['nutrient_focus']}")
        print(f"Urgency:        {rec['urgency']}")
        for s in rec['actions']:
            print(f"  * {s}")
    else:
        # Default action: extract features and train
        train_and_evaluate_models()


if __name__ == '__main__':
    main()
