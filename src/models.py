"""
Machine learning model definitions, training, and cross-validation for Cervical Cancer Risk Detection.
Provides Logistic Regression, Random Forest, and Support Vector Machine (SVM) classifiers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC


def get_model(
    model_name: str,
    class_weight: Optional[Union[str, Dict]] = "balanced",
    random_state: int = 42,
    **kwargs: Any
) -> Any:
    """
    Factory function to instantiate models with clinical hyperparameter defaults.

    Supported model names:
    - 'logistic_regression' or 'lr'
    - 'random_forest' or 'rf'
    - 'svm' or 'svc'
    - 'gradient_boosting' or 'gb'
    """
    key = model_name.lower().strip()

    if key in ["logistic_regression", "lr", "logistic regression"]:
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            solver="lbfgs",
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=class_weight,
            random_state=random_state
        )

    elif key in ["random_forest", "rf", "random forest"]:
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 6),
            min_samples_split=kwargs.get("min_samples_split", 4),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            class_weight=class_weight,
            random_state=random_state
        )

    elif key in ["svm", "svc", "support vector machine"]:
        return SVC(
            C=kwargs.get("C", 1.0),
            kernel=kwargs.get("kernel", "rbf"),
            gamma=kwargs.get("gamma", "scale"),
            probability=True,  # Crucial for Platt scaling risk probability
            class_weight=class_weight,
            random_state=random_state
        )

    elif key in ["gradient_boosting", "gb", "gradient boosting"]:
        return GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 3),
            random_state=random_state
        )

    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported: 'logistic_regression', 'random_forest', 'svm', 'gradient_boosting'"
        )


def run_stratified_cv(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Execute Stratified K-Fold cross-validation and return mean scores.
    Uses balanced scoring metrics tailored to clinical imbalanced screening.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision"
    }

    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        error_score="raise"
    )

    summary = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
        "cv_precision_std": float(np.std(cv_results["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
        "cv_recall_std": float(np.std(cv_results["test_recall"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_f1_std": float(np.std(cv_results["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_results["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_results["test_average_precision"]))
    }

    return summary


def get_default_models_suite() -> Dict[str, Any]:
    """Return dictionary of all three core models configured with default settings."""
    return {
        "Logistic Regression": get_model("lr", class_weight="balanced"),
        "Random Forest": get_model("rf", class_weight="balanced"),
        "Support Vector Machine": get_model("svm", class_weight="balanced")
    }
