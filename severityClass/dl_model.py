"""
severityClass/dl_model.py — Deep Learning Severity Classifier & Grad-CAM Interpretability.

Uses MobileNetV2 transfer learning with botanical image augmentation to classify leaf images
into severity grades (Healthy, Mild, Moderate, Severe).
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) on the final convolutional layer
to visualize the exact spatial regions (chlorotic patches vs vein architecture) driving the network's decisions.
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from severityClass.recommendations import get_recommendation

SEVERITY_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(SEVERITY_DIR, 'models')
GRADCAM_DIR = os.path.join(SEVERITY_DIR, 'gradcam')
FEATURES_CSV_PATH = os.path.join(SEVERITY_DIR, 'severity_features.csv')

SEVERITY_CLASSES = ['Healthy', 'Mild', 'Moderate', 'Severe']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SEVERITY_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(SEVERITY_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with Horticultural Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class LeafSeverityDataset(Dataset):
    """PyTorch Dataset for Rosa-sinensis leaf images with data augmentation."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label_str = row['severity_class']
        label_idx = CLASS_TO_IDX.get(label_str, 0)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label_idx, row['image_id'], image_path


def get_transforms():
    """Return train and validation transform pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV2 with a custom 4-class severity classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Implementation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for MobileNetV2.
    Computes activation heatmaps from the last convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for a target class."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to retain only positive influence
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cv2.resize(cam, (224, 224))


def generate_gradcam_overlay(image_bgr: np.ndarray, heatmap: np.ndarray,
                             title: str, conf: float) -> np.ndarray:
    """Blend Grad-CAM heatmap with original BGR image and draw informational banner."""
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Colorize heatmap (JET colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # 55% original, 45% heatmap
    blended = cv2.addWeighted(image_bgr, 0.55, heatmap_colored, 0.45, 0)

    # Draw header banner
    cv2.rectangle(blended, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.rectangle(blended, (0, 56), (w, 60), (0, 215, 255), -1)
    text = f"{title} (Confidence: {conf:.1%})"
    cv2.putText(blended, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_dl_model(num_epochs: int = 25, batch_size: int = 4, lr: float = 1e-4) -> Dict[str, Any]:
    """
    Train MobileNetV2 on the dataset using transfer learning with class weighting.
    """
    if not os.path.exists(FEATURES_CSV_PATH):
        raise FileNotFoundError(
            f"Features CSV not found at {FEATURES_CSV_PATH}. Run 'python -m severityClass.ml_model --train' first."
        )

    df = pd.read_csv(FEATURES_CSV_PATH)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Deep Learning Severity Classifier (MobileNetV2 Transfer Learning)")
    print("=" * 70)
    print(f"Total dataset samples: {len(df)}")
    counts = df['severity_class'].value_counts()
    for cls in SEVERITY_CLASSES:
        print(f"  - {cls:10s}: {counts.get(cls, 0):2d} sample(s)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    # Compute inverse class frequency weights
    total_samples = len(df)
    class_weights = []
    for cls in SEVERITY_CLASSES:
        c = counts.get(cls, 0)
        weight = total_samples / (len(SEVERITY_CLASSES) * max(c, 1))
        class_weights.append(weight)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_transform, val_transform = get_transforms()
    train_dataset = LeafSeverityDataset(df, transform=train_transform)
    eval_dataset = LeafSeverityDataset(df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model = build_mobilenetv2_model(num_classes=len(SEVERITY_CLASSES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    model.train()
    print(f"\nTraining MobileNetV2 for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:2d}/{num_epochs:2d}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1%}")

    # Save checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, 'mobilenetv2_severity.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved model weights to {checkpoint_path}")

    # Evaluation on full dataset
    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    sample_records = []

    with torch.no_grad():
        for images, labels, img_id, img_paths in eval_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

            y_true.append(IDX_TO_CLASS[labels.item()])
            y_pred.append(IDX_TO_CLASS[pred_idx])
            probabilities.append(probs)
            sample_records.append({
                'image_id': img_id[0],
                'path': img_paths[0],
                'true_label': IDX_TO_CLASS[labels.item()],
                'pred_label': IDX_TO_CLASS[pred_idx],
                'confidence': float(probs[pred_idx])
            })

    acc = accuracy_score(y_true, y_pred)
    present_classes = [c for c in SEVERITY_CLASSES if c in np.unique(y_true)]
    report_text = classification_report(y_true, y_pred, labels=present_classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    print("\n" + "-" * 70)
    print(f"MobileNetV2 Evaluation Accuracy: {acc:.1%}")
    print("-" * 70)
    print("Classification Report:")
    print(report_text)
    print("Confusion Matrix:")
    header = f"{'':12s}" + "".join([f"{c:>10s}" for c in present_classes])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{present_classes[i]:12s}" + "".join([f"{val:>10d}" for val in row])
        print(row_str)

    # Plot confusion matrix
    cm_path = os.path.join(MODELS_DIR, 'confusion_matrix_dl.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=present_classes,
        yticklabels=present_classes,
        title="MobileNetV2 Severity Confusion Matrix",
        ylabel="True Severity Class",
        xlabel="Predicted Severity Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    fig.tight_layout()
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    # Generate Grad-CAM for one representative image per class
    print("\nGenerating Grad-CAM visual heatmaps for each severity class...")
    target_layer = model.features[-1]  # Last inverted residual block
    gradcam = GradCAM(model, target_layer)

    gradcam_paths = {}
    classes_covered = set()

    for rec in sample_records:
        true_cls = rec['true_label']
        if true_cls in classes_covered:
            continue

        raw_bgr = cv2.imread(rec['path'])
        if raw_bgr is None:
            continue

        pil_img = Image.open(rec['path']).convert('RGB')
        tensor_img = val_transform(pil_img).unsqueeze(0).to(device)

        heatmap = gradcam.generate_heatmap(tensor_img, class_idx=CLASS_TO_IDX[true_cls])
        overlay = generate_gradcam_overlay(
            raw_bgr, heatmap,
            f"Grad-CAM: {true_cls} (Leaf #{rec['image_id']})",
            rec['confidence']
        )

        out_path = os.path.join(GRADCAM_DIR, f"gradcam_{true_cls.lower()}.jpg")
        cv2.imwrite(out_path, overlay)
        gradcam_paths[true_cls] = out_path
        classes_covered.add(true_cls)
        print(f"  -> Generated {true_cls:10s} Grad-CAM: {out_path}")

        if len(classes_covered) == len(present_classes):
            break

    return {
        'accuracy': acc,
        'report_text': report_text,
        'confusion_matrix': cm,
        'gradcam_paths': gradcam_paths,
        'sample_records': sample_records,
    }


def predict_leaf_dl(image_path: str, generate_cam: bool = False) -> Dict[str, Any]:
    """
    Predict severity class of an individual leaf using MobileNetV2 and optionally generate Grad-CAM.
    """
    checkpoint_path = os.path.join(MODELS_DIR, 'mobilenetv2_severity.pth')
    if not os.path.exists(checkpoint_path):
        train_dl_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_mobilenetv2_model(num_classes=len(SEVERITY_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    _, val_transform = get_transforms()
    pil_img = Image.open(image_path).convert('RGB')
    tensor_img = val_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = IDX_TO_CLASS[pred_idx]
        conf = float(probs[pred_idx])

    cam_path = None
    if generate_cam:
        target_layer = model.features[-1]
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate_heatmap(tensor_img, class_idx=pred_idx)
        raw_bgr = cv2.imread(image_path)
        overlay = generate_gradcam_overlay(
            raw_bgr, heatmap, f"MobileNetV2: {pred_class}", conf
        )
        os.makedirs(GRADCAM_DIR, exist_ok=True)
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        cam_path = os.path.join(GRADCAM_DIR, f"gradcam_{img_id}.jpg")
        cv2.imwrite(cam_path, overlay)

    rec = get_recommendation(pred_class)
    prob_dict = {cls: float(probs[idx]) for idx, cls in enumerate(SEVERITY_CLASSES)}

    return {
        'image_path': image_path,
        'predicted_class': pred_class,
        'confidence': conf,
        'probabilities': prob_dict,
        'gradcam_path': cam_path,
        'recommendation': rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Severity Classifier (MobileNetV2 Transfer Learning)")
    parser.add_argument('--train', action='store_true', help="Train MobileNetV2 and generate Grad-CAMs")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--predict', type=str, default=None, help="Predict severity for a single leaf image")
    parser.add_argument('--gradcam', action='store_true', help="Generate Grad-CAM heatmap for prediction")
    args = parser.parse_args()

    if args.predict:
        res = predict_leaf_dl(args.predict, generate_cam=args.gradcam)
        print("=" * 70)
        print("  MobileNetV2 Severity Prediction")
        print("=" * 70)
        print(f"Image:           {args.predict}")
        print(f"Predicted Class: {res['predicted_class']} (Confidence: {res['confidence']:.1%})")
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
        if res.get('gradcam_path'):
            print(f"\nSaved Grad-CAM visualization: {res['gradcam_path']}")
    else:
        train_dl_model(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
