"""
Data preprocessing and leakage-prevention pipeline for Cervical Cancer Risk Prediction.
Ensures imputation statistics, feature scaling, and resampling are strictly fit on training data.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class CervicalDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free preprocessor that:
    1. Removes features with missing rates exceeding a configured threshold (e.g. >50%).
    2. Imputes missing values using training-set medians (numeric) or modes (binary/categorical).
    3. Standardizes/scales features using a chosen scaler fit strictly on training records.
    """

    def __init__(
        self,
        missing_threshold: float = 0.50,
        scaling_method: Literal["standard", "robust", "minmax", "none"] = "standard",
        impute_strategy: Literal["median", "mean"] = "median"
    ):
        self.missing_threshold = missing_threshold
        self.scaling_method = scaling_method
        self.impute_strategy = impute_strategy

        # Fitted state
        self.dropped_columns_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.impute_values_: Dict[str, float] = {}
        self.scaler_: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CervicalDataPreprocessor":
        """Fit preprocessor state on training features without leakage."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)

        # 1. Identify columns exceeding missingness threshold
        missing_rates = X.isna().mean()
        self.dropped_columns_ = missing_rates[missing_rates > self.missing_threshold].index.tolist()

        remaining_cols = [c for c in self.feature_names_in_ if c not in self.dropped_columns_]
        self.feature_names_out_ = remaining_cols

        # 2. Compute training-set imputation values
        self.impute_values_ = {}
        for col in remaining_cols:
            series = X[col].dropna()
            if len(series) == 0:
                val = 0.0
            elif series.nunique() <= 2:
                # Binary/categorical indicator -> use mode
                val = float(series.mode().iloc[0]) if not series.mode().empty else 0.0
            else:
                # Numeric continuous -> median or mean
                val = float(series.median()) if self.impute_strategy == "median" else float(series.mean())
            self.impute_values_[col] = val

        # 3. Fit scaler on imputed training data
        X_imputed = self._impute_dataframe(X[remaining_cols])
        if self.scaling_method == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_imputed)
        elif self.scaling_method == "robust":
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X_imputed)
        elif self.scaling_method == "minmax":
            self.scaler_ = MinMaxScaler()
            self.scaler_.fit(X_imputed)
        else:
            self.scaler_ = None

        return self

    def _impute_dataframe(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """Helper to fill missing values using fitted impute values."""
        imputed = df_subset.copy()
        for col, val in self.impute_values_.items():
            if col in imputed.columns:
                imputed[col] = imputed[col].fillna(val)
            else:
                # Fill missing column if absent in unseen data with training default
                imputed[col] = val
        return imputed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform unseen data using strictly training-fitted parameters."""
        if not self.feature_names_out_:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_[:X.shape[1]])

        # Drop columns identified during training
        cols_to_keep = [c for c in self.feature_names_out_ if c in X.columns]
        missing_expected_cols = [c for c in self.feature_names_out_ if c not in X.columns]

        X_subset = X[cols_to_keep].copy()

        # Add any missing expected columns with fitted default
        for c in missing_expected_cols:
            X_subset[c] = self.impute_values_.get(c, 0.0)

        # Reorder to match exact training column sequence
        X_subset = X_subset[self.feature_names_out_]

        # Impute
        X_imputed = self._impute_dataframe(X_subset)

        # Scale
        if self.scaler_ is not None:
            scaled_array = self.scaler_.transform(X_imputed)
            return pd.DataFrame(scaled_array, columns=self.feature_names_out_, index=X.index)
        
        return X_imputed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training features."""
        return self.fit(X, y).transform(X)


def split_data_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split ensuring minority class prevalence
    is preserved across both sets without data leakage.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: Literal["smote", "random_oversample", "random_undersample", "none"] = "smote",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample the training dataset to mitigate severe clinical class imbalance.
    NOTE: Must ONLY be called on training data, never on the holdout test set.
    """
    if strategy == "none" or not IMBLEARN_AVAILABLE:
        return X_train, y_train

    # Check minority class count for SMOTE k_neighbors
    minority_count = y_train.value_counts().min()
    if minority_count < 2:
        # Too few positive instances to resample safely
        return X_train, y_train

    k_neighbors = min(5, minority_count - 1)

    if strategy == "smote":
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    elif strategy == "random_oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == "random_undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        return X_train, y_train

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Reconstruct DataFrame with preserved column names
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)

    return X_resampled, y_resampled
