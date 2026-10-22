"""
Unit tests for pipeline orchestration, saving/loading, and inference.
"""

import os
import pandas as pd
import pytest
from src.data_loader import load_dataset
from src.pipeline import CervicalCancerPipeline


def test_pipeline_train_evaluate_and_predict():
    """Test full pipeline lifecycle on a slice of the dataset."""
    df = load_dataset().head(120)  # fast slice for unit test
    # Ensure there are at least a couple of positive samples
    df.loc[0:5, "Biopsy"] = 1
    df.loc[6:, "Biopsy"] = 0

    pipe = CervicalCancerPipeline(
        model_name="random_forest",
        target_column="Biopsy",
        imbalance_strategy="random_oversample",
        random_state=42
    )

    results = pipe.train_and_evaluate(df, test_size=0.25, model_kwargs={"n_estimators": 10, "max_depth": 3})

    assert "test_metrics" in results
    assert "accuracy" in results["test_metrics"]
    assert "sensitivity" in results["test_metrics"]

    # Test single patient prediction
    sample_patient = {col: 0 for col in pipe.feature_names_in_}
    sample_patient["Age"] = 40
    sample_patient["Smokes"] = 1
    sample_patient["Dx:HPV"] = 1

    single_pred = pipe.predict_single_patient(sample_patient)
    assert "predicted_probability" in single_pred
    assert 0.0 <= single_pred["predicted_probability"] <= 1.0
    assert "risk_tier" in single_pred
    assert "uncertainty_level" in single_pred


def test_pipeline_serialization_and_deserialization(tmp_path):
    """Test joblib save and reload preserves prediction consistency."""
    df = load_dataset().head(100)
    df.loc[0:4, "Biopsy"] = 1
    df.loc[5:, "Biopsy"] = 0

    pipe = CervicalCancerPipeline(model_name="logistic_regression", target_column="Biopsy", imbalance_strategy="none")
    pipe.train_and_evaluate(df, test_size=0.25)

    save_path = str(tmp_path / "model_test.joblib")
    pipe.save(save_path)
    assert os.path.exists(save_path)

    loaded_pipe = CervicalCancerPipeline.load(save_path)
    test_record = pd.DataFrame([{c: 0 for c in pipe.feature_names_in_}])

    orig_proba = pipe.predict_proba(test_record)
    loaded_proba = loaded_pipe.predict_proba(test_record)

    assert abs(orig_proba[0] - loaded_proba[0]) < 1e-6
