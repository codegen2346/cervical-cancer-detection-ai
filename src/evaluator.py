"""
Evaluation module for Cervical Cancer Risk Detection models.
Calculates clinical metrics (Sensitivity, Specificity, F1, ROC-AUC, PR-AUC)
and generates diagnostic visualizations (Confusion Matrices, ROC curves, PR curves, Threshold tradeoffs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive medical classification metrics.
    If y_proba is provided, y_pred is computed using the custom decision threshold.
    """
    y_true_arr = np.asarray(y_true).astype(int)

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        # Apply custom decision threshold
        y_pred_arr = (y_proba_arr >= threshold).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred).astype(int)
        y_proba_arr = None

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Clinical metrics
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    sensitivity = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))  # True Positive Rate
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0               # True Negative Rate
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true_arr)),
        "prevalence": float(np.mean(y_true_arr))
    }

    # Probability-based metrics (ROC-AUC & PR-AUC)
    if y_proba_arr is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        except ValueError:
            metrics["roc_auc"] = 0.5

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Classifier",
    target_name: str = "Biopsy Outcome"
) -> plt.Figure:
    """Generate a medical-grade confusion matrix heatmap with counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    total = np.sum(cm)
    percentages = (cm / total * 100).round(1)

    labels = [
        [f"True Neg (TN)\n{cm[0, 0]}\n({percentages[0, 0]}%)", f"False Pos (FP)\n{cm[0, 1]}\n({percentages[0, 1]}%)"],
        [f"False Neg (FN)\n{cm[1, 0]}\n({percentages[1, 0]}%)", f"True Pos (TP)\n{cm[1, 1]}\n({percentages[1, 1]}%)"]
    ]

    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "medium"}
    )

    ax.set_title(f"Confusion Matrix - {model_name}\nTarget: {target_name}", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Outcome", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Clinical Outcome", fontsize=11, labelpad=8)
    plt.tight_layout()
    return fig


def plot_roc_curve(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Receiver Operating Characteristic (ROC) Comparison"
) -> plt.Figure:
    """
    Plot ROC curves for one or multiple models.
    results_dict maps model_name -> (y_true, y_proba).
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)

    for model_name, (y_true, y_proba) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall (PR) Curve Comparison"
) -> plt.Figure:
    """
    Plot PR curves for one or multiple models. Crucial for imbalanced medical cohorts.
    results_dict maps model_name -> (y_true, y_proba).
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)

    for model_name, (y_true, y_proba) in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})")

    # Plot baseline prevalence
    first_y_true = list(results_dict.values())[0][0]
    prevalence = np.mean(first_y_true)
    ax.axhline(prevalence, color="grey", linestyle="--", lw=1.5, label=f"No Skill Baseline ({prevalence:.1%})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall / Sensitivity", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    return fig


def compute_threshold_tradeoffs(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    n_points: int = 50
) -> pd.DataFrame:
    """Compute sensitivity, specificity, precision, and F1 across decision thresholds."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba)

    thresholds = np.linspace(0.05, 0.95, n_points)
    rows = []

    for t in thresholds:
        preds = (y_proba_arr >= t).astype(int)
        cm = confusion_matrix(y_true_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 3),
            "sensitivity": round(float(sens), 3),
            "specificity": round(float(spec), 3),
            "precision": round(float(prec), 3),
            "f1": round(float(f1), 3),
            "false_negatives": int(fn),
            "false_positives": int(fp)
        })

    return pd.DataFrame(rows)


def plot_threshold_tradeoff_curve(tradeoff_df: pd.DataFrame, current_threshold: float = 0.5) -> plt.Figure:
    """Plot Sensitivity, Specificity, Precision, and F1 as a function of classification threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=120)

    ax.plot(tradeoff_df["threshold"], tradeoff_df["sensitivity"], label="Sensitivity / Recall (TPR)", color="#d9534f", lw=2.2)
    ax.plot(tradeoff_df["threshold"], tradeoff_df["specificity"], label="Specificity (TNR)", color="#0275d8", lw=2.2)
    ax.plot(tradeoff_df["threshold"], tradeoff_df["precision"], label="Precision (PPV)", color="#5cb85c", lw=1.8, linestyle="-.")
    ax.plot(tradeoff_df["threshold"], tradeoff_df["f1"], label="F1 Score", color="#f0ad4e", lw=1.8, linestyle=":")

    ax.axvline(current_threshold, color="black", linestyle="--", lw=1.5, label=f"Selected Cutoff ({current_threshold:.2f})")

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Clinical Threshold Tuning: Sensitivity vs. Specificity Trade-Off", fontsize=12, pad=12)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="center left", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    return fig


def classify_risk_tier(probability: float) -> Tuple[str, str, str]:
    """
    Categorize predicted risk into clinical research tiers:
    - Low Risk: p < 0.20
    - Moderate Risk: 0.20 <= p < 0.50
    - Elevated / High Risk: p >= 0.50
    Returns (tier_label, hex_color, advisory_text).
    """
    if probability < 0.20:
        return (
            "Low Risk Profile",
            "#28a745",
            "Screening indicators are largely concordant with baseline population norms. Routine guideline-based screening intervals recommended."
        )
    elif probability < 0.50:
        return (
            "Moderate / Borderline Risk Profile",
            "#ffc107",
            "Elevated risk signals detected from co-factors (e.g. contraception duration, smoking, or history). Recommended for closer clinical review and standard follow-up."
        )
    else:
        return (
            "Elevated Risk / Action Indicated",
            "#dc3545",
            "Notable risk factor convergence. In clinical research protocols, this warrants diagnostic triaging (e.g., colposcopy referral or cytology review). This is an experimental model output and NOT a diagnosis."
        )
