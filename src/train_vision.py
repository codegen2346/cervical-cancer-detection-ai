"""
Automated Training Engine for Real Clinical Cervical Cytology Images (Herlev Hospital Dataset).
Trains the deep learning vision model with automated augmentation, class weighting,
and comprehensive clinical evaluation (Accuracy, Sensitivity, Specificity, ROC-AUC).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.image_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CervicalEfficientNetB0,
    CervicalTransferCNN,
)


def train_cervical_vision_model(
    data_dir: str,
    output_model_path: str,
    model_architecture: str = "mobilenet_v3",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.0005,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Automated training and evaluation routine for Cervical Cytology Vision Models.
    Supports both MobileNetV3-Small and EfficientNet-B0 architectures.
    
    1. Loads train/test image folders (Herlev University Hospital benchmark).
    2. Applies medical rotational and contrast augmentations.
    3. Handles class imbalance via balanced loss weighting.
    4. Trains CNN with Cosine Annealing learning rate.
    5. Evaluates on holdout test set with clinical metrics (Accuracy, Sensitivity, Specificity, ROC-AUC).
    6. Saves model checkpoint and comprehensive metrics JSON.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Training and test directories expected in {data_dir}")

    # 1. Automatic Medical Data Augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=90),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset classes: {train_dataset.class_to_idx} (0=Abnormal/Cancer, 1=Normal)")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 2. Balanced Class Weighting
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Loss class weights: {weight_tensor.tolist()}")

    # 3. Model Architecture Selection
    is_eff = "efficientnet" in model_architecture.lower()
    arch_key = "efficientnet_b0" if is_eff else "mobilenet_v3"
    display_title = "EfficientNet-B0" if is_eff else "MobileNetV3-Small"

    if is_eff:
        model = CervicalEfficientNetB0(num_classes=2, pretrained=True).to(device)
    else:
        model = CervicalTransferCNN(num_classes=2, pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized {display_title} with {total_params:,} parameters.")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_metrics = {}
    start_time = time.time()

    print(f"Beginning automated fine-tuning for {display_title} ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)

        # 4. Evaluation on Holdout Test Set
        model.eval()
        all_preds = []
        all_cancer_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                # Softmax: col 0 is abnormal/cancer, col 1 is normal
                probs = torch.softmax(outputs, dim=1)
                cancer_p = probs[:, 0]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_cancer_probs.extend(cancer_p.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_cancer_proba = np.array(all_cancer_probs)

        # Class 0: Cancer / Abnormal, Class 1: Normal / Benign
        acc = float(accuracy_score(y_true, y_pred))

        # Invert labels for Cancer-focused precision & recall (0=Cancer as positive class)
        y_true_cancer = (y_true == 0).astype(int)
        y_pred_cancer = (y_pred == 0).astype(int)

        cancer_sens = float(recall_score(y_true_cancer, y_pred_cancer, zero_division=0))
        cancer_prec = float(precision_score(y_true_cancer, y_pred_cancer, zero_division=0))
        cancer_f1 = float(f1_score(y_true_cancer, y_pred_cancer, zero_division=0))

        # Normal specificity: TN for cancer / (TN + FP)
        cm_cancer = confusion_matrix(y_true_cancer, y_pred_cancer, labels=[0, 1])
        # [0, 0] = TN (Normal correctly identified as normal)
        # [0, 1] = FP (Normal falsely identified as cancer)
        # [1, 0] = FN (Cancer missed)
        # [1, 1] = TP (Cancer detected)
        tn = int(cm_cancer[0, 0])
        fp = int(cm_cancer[0, 1])
        fn = int(cm_cancer[1, 0])
        tp = int(cm_cancer[1, 1])
        benign_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        try:
            auc = float(roc_auc_score(y_true_cancer, y_cancer_proba))
        except Exception:
            auc = 0.5

        print(
            f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} | "
            f"Val Acc: {acc:.1%} | Cancer Sens: {cancer_sens:.1%} | "
            f"Spec: {benign_spec:.1%} | F1: {cancer_f1:.3f} | AUC: {auc:.3f}"
        )

        composite_score = (acc * 0.4) + (cancer_f1 * 0.3) + (auc * 0.3)
        if composite_score > best_score or epoch == epochs:
            best_score = composite_score
            best_metrics = {
                "architecture_key": arch_key,
                "display_title": display_title,
                "epoch": epoch,
                "accuracy": round(acc, 4),
                "sensitivity": round(cancer_sens, 4),
                "specificity": round(benign_spec, 4),
                "precision": round(cancer_prec, 4),
                "f1": round(cancer_f1, 4),
                "roc_auc": round(auc, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "total_parameters": total_params,
                "training_time_sec": round(time.time() - start_time, 1)
            }
            os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)

    # 5. Benchmark Latency Test (Average CPU inference latency in milliseconds)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(latencies)), 2)
    best_metrics["avg_latency_ms"] = avg_latency_ms

    file_size_mb = round(os.path.getsize(output_model_path) / (1024 * 1024), 2)
    best_metrics["model_size_mb"] = file_size_mb

    # 6. Save Metrics JSON for Dashboard
    metrics_path = os.path.join(os.path.dirname(output_model_path), f"{arch_key}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nSaved {display_title} checkpoint to: {output_model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Metrics: {best_metrics}")
    return best_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Cervical Vision Deep Learning Models")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["mobilenet_v3", "efficientnet_b0", "both"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images", "real_herlev"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    if args.model in ["mobilenet_v3", "both"]:
        print("\n================== TRAINING MOBILENET-V3-SMALL ==================")
        mob_path = os.path.join(models_dir, "cervical_vision_model.pth")
        train_cervical_vision_model(base_dir, mob_path, model_architecture="mobilenet_v3", epochs=args.epochs)

    if args.model in ["efficientnet_b0", "both"]:
        print("\n================== TRAINING EFFICIENTNET-B0 ==================")
        eff_path = os.path.join(models_dir, "cervical_efficientnet_b0.pth")
        train_cervical_vision_model(base_dir, eff_path, model_architecture="efficientnet_b0", epochs=args.epochs)
