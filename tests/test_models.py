"""
Unit tests for models and cross-validation module.
"""

import numpy as np
import pytest
from src.models import get_default_models_suite, get_model, run_stratified_cv


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = np.array([1 if i < 15 else 0 for i in range(80)]) # Imbalanced
    return X, y


def test_get_models_instantiation():
    """Verify all 3 core models are properly constructed."""
    lr = get_model("lr")
    rf = get_model("rf")
    svm = get_model("svm")

    assert hasattr(lr, "fit")
    assert hasattr(rf, "fit")
    assert hasattr(svm, "fit")
    # Verify SVM has probability=True for posterior calibration
    assert svm.probability is True


def test_stratified_cv_runs_cleanly(dummy_train_data):
    """Verify Stratified CV computes expected metric keys."""
    X, y = dummy_train_data
    rf = get_model("rf", n_estimators=10, max_depth=3)
    cv_res = run_stratified_cv(rf, X, y, n_splits=3)

    assert "cv_accuracy_mean" in cv_res
    assert "cv_recall_mean" in cv_res
    assert "cv_roc_auc_mean" in cv_res
    assert 0.0 <= cv_res["cv_accuracy_mean"] <= 1.0


def test_models_suite_returns_dictionary():
    suite = get_default_models_suite()
    assert "Logistic Regression" in suite
    assert "Random Forest" in suite
    assert "Support Vector Machine" in suite
