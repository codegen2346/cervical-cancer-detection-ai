"""
Unit tests for data_loader module.
"""

import os
import pandas as pd
import pytest
from src.data_loader import (
    TARGET_COLUMNS,
    FEATURE_METADATA,
    generate_sample_patients_csv,
    get_default_data_dir,
    load_dataset,
    split_features_and_target,
)


def test_feature_metadata_validity():
    """Verify metadata covers expected risk features and has proper schema."""
    assert "Age" in FEATURE_METADATA
    assert "Smokes" in FEATURE_METADATA
    assert "Dx:HPV" in FEATURE_METADATA
    assert FEATURE_METADATA["Age"]["type"] == "numeric"
    assert FEATURE_METADATA["Smokes"]["type"] == "binary"


def test_load_dataset_cached():
    """Ensure dataset loads as DataFrame and converts '?' to NaN."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 800
    assert "Biopsy" in df.columns
    # Check that '?' was converted to float/numeric
    assert pd.api.types.is_numeric_dtype(df["Age"])


def test_split_features_and_target_prevents_leakage():
    """Ensure split_features_and_target removes all target columns from X by default."""
    df = load_dataset()
    X, y = split_features_and_target(df, target_column="Biopsy", include_other_screenings_as_features=False)

    assert "Biopsy" not in X.columns
    assert "Hinselmann" not in X.columns
    assert "Schiller" not in X.columns
    assert "Citology" not in X.columns
    assert y.name == "Biopsy"
    assert len(X) == len(y)


def test_generate_sample_patients_csv(tmp_path):
    """Ensure sample CSV generator creates valid patient records."""
    out_file = str(tmp_path / "test_sample_patients.csv")
    res_path = generate_sample_patients_csv(out_file)
    assert os.path.exists(res_path)

    sample_df = pd.read_csv(res_path)
    assert len(sample_df) == 5
    assert "Patient_ID" in sample_df.columns
    assert "Age" in sample_df.columns
