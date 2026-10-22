"""
Model interpretability and explainability module for Cervical Cancer Risk Detection.
Provides SHAP values, feature importance rankings, and local patient attribution explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract global feature importance or coefficient weights from fitted models.
    Supports RandomForestClassifier, GradientBoostingClassifier, and LogisticRegression.
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based model (MDI Gini importance)
        scores = model.feature_importances_
        metric_name = "Importance (MDI)"
    elif hasattr(model, "coef_"):
        # Linear model (Logistic Regression coefficients / log-odds)
        scores = np.abs(model.coef_[0])
        metric_name = "Abs Coefficient (|log-odds|)"
    else:
        # Fallback uniform
        scores = np.ones(len(feature_names)) / len(feature_names)
        metric_name = "Uniform"

    df = pd.DataFrame({
        "feature": feature_names,
        "score": scores
    }).sort_values(by="score", ascending=False).reset_index(drop=True)
    df["metric"] = metric_name
    return df


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Top Predictive Clinical Risk Factors"
) -> plt.Figure:
    """Render a horizontal bar plot of the top predictive features."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    subset = importance_df.head(top_n).iloc[::-1]  # Invert so top feature is on top

    bars = ax.barh(subset["feature"], subset["score"], color="#337ab7", edgecolor="#23527c", height=0.65)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Add numeric labels to bars
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9
        )

    plt.tight_layout()
    return fig


def compute_shap_explanations(
    model: Any,
    X_train_sample: pd.DataFrame,
    X_eval: pd.DataFrame
) -> Optional[Any]:
    """
    Compute SHAP Explanation object using TreeExplainer, LinearExplainer, or Explainer.
    Subsamples background data for performance and stability.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        if hasattr(model, "estimators_"):  # Random Forest / Gradient Boosting
            explainer = shap.TreeExplainer(model, data=shap.sample(X_train_sample, 50, random_state=42))
            shap_values = explainer(X_eval)
        elif hasattr(model, "coef_"):      # Logistic Regression
            explainer = shap.LinearExplainer(model, shap.sample(X_train_sample, 50, random_state=42))
            shap_values = explainer(X_eval)
        else:
            explainer = shap.Explainer(model.predict_proba, shap.sample(X_train_sample, 50, random_state=42))
            shap_values = explainer(X_eval)

        return shap_values
    except Exception as e:
        print(f"SHAP explanation computation error: {e}")
        return None


def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    max_display: int = 12
) -> Optional[plt.Figure]:
    """Generate SHAP beeswarm / summary plot."""
    if not SHAP_AVAILABLE or shap_values is None:
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        # If binary classification output has 2 slices, select positive class (1)
        values_to_plot = shap_values[..., 1] if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2 else shap_values

        shap.plots.beeswarm(values_to_plot, max_display=max_display, show=False)
        plt.title("SHAP Global Impact on Cervical Cancer Risk Outcome", fontsize=12, pad=12)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
        return None


def plot_patient_risk_waterfall(
    model: Any,
    patient_series: pd.Series,
    feature_names: List[str],
    top_n: int = 10
) -> plt.Figure:
    """
    Patient-level risk driver visualization.
    Shows the top features that contributed positively or negatively to the risk estimation.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)

    # Estimate contribution via feature values * model weights/importances
    if hasattr(model, "coef_"):
        weights = model.coef_[0]
        contributions = patient_series.values[:len(weights)] * weights
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
        # Directional heuristic: positive values relative to standard mean
        contributions = patient_series.values[:len(weights)] * weights
    else:
        contributions = patient_series.values

    contrib_df = pd.DataFrame({
        "feature": feature_names[:len(contributions)],
        "value": patient_series.values[:len(contributions)],
        "impact": contributions
    })

    # Sort by absolute impact
    contrib_df["abs_impact"] = contrib_df["impact"].abs()
    top_contrib = contrib_df.sort_values(by="abs_impact", ascending=False).head(top_n).iloc[::-1]

    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in top_contrib["impact"]]

    bars = ax.barh(top_contrib["feature"], top_contrib["impact"], color=colors, height=0.6)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
    ax.set_xlabel("Directional Risk Impact (Red = Increases Risk, Green = Lowers/Protective)", fontsize=10)
    ax.set_title("Individual Patient Risk Drivers (Feature Attribution)", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()
    return fig
