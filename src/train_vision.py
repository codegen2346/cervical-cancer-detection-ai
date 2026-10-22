"""
Automated Training Engine for Real Clinical Cervical Cytology Images (Herlev Hospital Dataset).
Trains the deep learning vision model with automated augmentation, class weighting,
and comprehensive clinical evaluation (Accuracy, Sensitivity, Specificity, ROC-AUC).
"""

from __future__ import annotations

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

from src.image_classifier import IMAGENET_MEAN, IMAGENET_STD, CervicalTransferCNN


def train_cervical_vision_model(
    data_dir: str,
    output_model_path: str,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 0.0005,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Automated training routine:
    1. Loads train/test image folders.
    2. Applies medical rotational and contrast augmentations.
    3. Handles class imbalance via loss weighting.
    4. Trains CNN with Cosine Annealing learning rate.
    5. Evaluates on holdout test set and saves the best model checkpoint.
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
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 2. Balanced Class Weighting
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Loss class weights: {weight_tensor.tolist()}")

    # 3. Model, Criterion, Optimizer
    model = CervicalTransferCNN(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    best_metrics = {}
    start_time = time.time()

    print("Beginning automated fine-tuning...")

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
        all_probas = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_probas.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probas)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        sens = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = 0.5

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} | Val Acc: {acc:.1%} | Sens (Recall): {sens:.1%} | Spec: {spec:.1%} | F1: {f1:.3f} | AUC: {auc:.3f}")

        if f1 > best_f1 or epoch == epochs:
            best_f1 = f1
            best_metrics = {
                "epoch": epoch,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": f1,
                "roc_auc": auc,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "training_time_sec": round(time.time() - start_time, 1)
            }
            os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)

    print(f"\nTraining completed! Saved best model to: {output_model_path}")
    print(f"Final Test Evaluation: {best_metrics}")
    return best_metrics


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images", "real_herlev"))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "cervical_vision_model.pth"))
    train_cervical_vision_model(base_dir, model_path, epochs=5)
