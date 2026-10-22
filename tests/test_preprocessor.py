"""
Unit tests for preprocessor module and leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessor import (
    CervicalDataPreprocessor,
    handle_class_imbalance,
    split_data_stratified,
)


@pytest.fixture
def dummy_dataset():
    """Create synthetic dataset with missing values and severe class imbalance."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Age": np.random.randint(18, 60, size=n),
        "HighMissingCol": [np.nan if i > 10 else 1.0 for i in range(n)],  # ~90% missing
        "LowMissingCol": [np.nan if i < 5 else float(i % 3) for i in range(n)], # 5% missing
        "BinaryFlag": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "Target": [1 if i < 8 else 0 for i in range(n)] # 8% positive
    })
    return df


def test_preprocessor_drops_high_missing_columns(dummy_dataset):
    """Ensure columns exceeding threshold are dropped."""
    X = dummy_dataset.drop(columns=["Target"])
    preproc = CervicalDataPreprocessor(missing_threshold=0.50)
    preproc.fit(X)

    assert "HighMissingCol" in preproc.dropped_columns_
    assert "HighMissingCol" not in preproc.feature_names_out_
    assert "LowMissingCol" in preproc.feature_names_out_


def test_preprocessor_imputation_and_scaling(dummy_dataset):
    """Ensure no NaN values remain after transformation and scaling works."""
    X = dummy_dataset.drop(columns=["Target"])
    preproc = CervicalDataPreprocessor(missing_threshold=0.50, scaling_method="standard")
    X_proc = preproc.fit_transform(X)

    assert X_proc.isna().sum().sum() == 0
    # Mean of standard scaler on train should be ~0
    assert abs(X_proc["Age"].mean()) < 1e-6


def test_stratified_split_preserves_ratio(dummy_dataset):
    """Ensure stratified split maintains the minority class proportion."""
    X = dummy_dataset.drop(columns=["Target"])
    y = dummy_dataset["Target"]

    X_train, X_test, y_train, y_test = split_data_stratified(X, y, test_size=0.25, random_state=42)

    assert len(X_train) == 75
    assert len(X_test) == 25
    train_prev = y_train.mean()
    test_prev = y_test.mean()
    # Prevalence should be virtually identical
    assert abs(train_prev - test_prev) < 0.05


def test_smote_resampling_minority_balance(dummy_dataset):
    """Ensure SMOTE equalizes classes on training data without crashing."""
    X = dummy_dataset.drop(columns=["Target"])
    y = dummy_dataset["Target"]
    preproc = CervicalDataPreprocessor(missing_threshold=0.50)
    X_proc = preproc.fit_transform(X)

    X_res, y_res = handle_class_imbalance(X_proc, y, strategy="smote", random_state=42)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)
