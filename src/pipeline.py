"""
Unified Pipeline Orchestrator for Cervical Cancer Risk Detection Research.
Manages end-to-end data ingestion, preprocessing, resampling, model training,
cross-validation, serialization with joblib, and inference.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data_loader import TARGET_COLUMNS, load_dataset, split_features_and_target
from src.evaluator import classify_risk_tier, compute_metrics
from src.models import get_model, run_stratified_cv
from src.preprocessor import CervicalDataPreprocessor, handle_class_imbalance, split_data_stratified


class CervicalCancerPipeline:
    """
    Production-grade ML pipeline bundling:
    - Preprocessor (missingness filter, imputation, scaling)
    - Classifier (LR, RF, SVM)
    - Metadata, feature schema, threshold, and audit metrics
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        target_column: str = "Biopsy",
        imbalance_strategy: str = "smote",
        scaling_method: str = "standard",
        missing_threshold: float = 0.50,
        decision_threshold: float = 0.50,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.target_column = target_column
        self.imbalance_strategy = imbalance_strategy
        self.scaling_method = scaling_method
        self.missing_threshold = missing_threshold
        self.decision_threshold = decision_threshold
        self.random_state = random_state

        # Internal state
        self.preprocessor: Optional[CervicalDataPreprocessor] = None
        self.model: Optional[Any] = None
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.test_metrics_: Dict[str, float] = {}
        self.cv_metrics_: Dict[str, float] = {}
        self.training_time_iso_: str = ""
        self.dataset_shape_: Tuple[int, int] = (0, 0)

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end training and evaluation with strict leakage prevention:
        1. Stratified train/test split.
        2. Fit preprocessor strictly on train fold.
        3. Transform train fold & resample with SMOTE/class-imbalance handler.
        4. Transform test fold using train statistics (no resampling on test).
        5. Fit model.
        6. Compute holdout test metrics and cross-validation scores.
        """
        if model_kwargs is None:
            model_kwargs = {}

        self.dataset_shape_ = df.shape
        self.training_time_iso_ = datetime.datetime.now().isoformat()

        # Split features and target
        X, y = split_features_and_target(
            df,
            target_column=self.target_column,
            include_other_screenings_as_features=False
        )
        self.feature_names_in_ = list(X.columns)

        # 1. Stratified Train / Test Split
        X_train_raw, X_test_raw, y_train, y_test = split_data_stratified(
            X, y,
            test_size=test_size,
            random_state=self.random_state
        )

        # 2. Fit preprocessor ONLY on train fold
        self.preprocessor = CervicalDataPreprocessor(
            missing_threshold=self.missing_threshold,
            scaling_method=self.scaling_method,
            impute_strategy="median"
        )
        self.preprocessor.fit(X_train_raw)
        self.feature_names_out_ = self.preprocessor.feature_names_out_

        # 3. Transform train fold and handle class imbalance
        X_train_proc = self.preprocessor.transform(X_train_raw)
        X_train_resampled, y_train_resampled = handle_class_imbalance(
            X_train_proc,
            y_train,
            strategy=self.imbalance_strategy,
            random_state=self.random_state
        )

        # 4. Transform holdout test fold using fitted preprocessor
        X_test_proc = self.preprocessor.transform(X_test_raw)

        # 5. Initialize and train model
        self.model = get_model(
            self.model_name,
            class_weight="balanced" if self.imbalance_strategy == "none" else None,
            random_state=self.random_state,
            **model_kwargs
        )
        self.model.fit(X_train_resampled, y_train_resampled)

        # 6. Cross-Validation on training set
        try:
            self.cv_metrics_ = run_stratified_cv(
                self.model,
                X_train_resampled,
                y_train_resampled,
                n_splits=5,
                random_state=self.random_state
            )
        except Exception as e:
            self.cv_metrics_ = {"error": str(e)}

        # 7. Evaluate on holdout test set
        y_test_proba = self.model.predict_proba(X_test_proc)[:, 1]
        self.test_metrics_ = compute_metrics(
            y_true=y_test,
            y_pred=(y_test_proba >= self.decision_threshold).astype(int),
            y_proba=y_test_proba,
            threshold=self.decision_threshold
        )

        # Save test data for plotting in dashboard
        return {
            "test_metrics": self.test_metrics_,
            "cv_metrics": self.cv_metrics_,
            "y_test_true": y_test.values,
            "y_test_proba": y_test_proba,
            "X_test_proc": X_test_proc,
            "X_train_sample": X_train_proc.head(100)
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Transform unseen inputs and predict positive class probabilities."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Pipeline must be trained or loaded before prediction.")
        X_proc = self.preprocessor.transform(X)
        return self.model.predict_proba(X_proc)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary outcome using configured or custom threshold."""
        thresh = threshold if threshold is not None else self.decision_threshold
        proba = self.predict_proba(X)
        return (proba >= thresh).astype(int)

    def predict_single_patient(
        self,
        patient_data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Assess risk for a single patient record.
        Returns probability, binary outcome, research risk tier, and uncertainty metrics.
        """
        df_single = pd.DataFrame([patient_data])
        proba = float(self.predict_proba(df_single)[0])
        thresh = threshold if threshold is not None else self.decision_threshold
        pred = int(proba >= thresh)

        tier_label, hex_color, advisory = classify_risk_tier(proba)

        # Uncertainty indicator: closer to threshold means higher decision boundary uncertainty
        uncertainty_margin = abs(proba - thresh)
        if uncertainty_margin < 0.10:
            uncertainty_level = "High Decision Boundary Uncertainty (Marginal / Borderline)"
        elif uncertainty_margin < 0.25:
            uncertainty_level = "Moderate Decision Boundary Uncertainty"
        else:
            uncertainty_level = "Low Decision Boundary Uncertainty (High Model Separation)"

        return {
            "predicted_probability": round(proba, 4),
            "predicted_outcome": pred,
            "threshold_used": thresh,
            "risk_tier": tier_label,
            "tier_color": hex_color,
            "clinical_advisory": advisory,
            "uncertainty_level": uncertainty_level,
            "target_evaluated": self.target_column,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Score a batch DataFrame of patient records and append risk assessment columns."""
        thresh = threshold if threshold is not None else self.decision_threshold
        df_scored = df_batch.copy()

        # Separate identifiers if present
        id_cols = [c for c in ["Patient_ID", "ID", "patient_id", "id"] if c in df_scored.columns]

        # Features subset
        feature_cols = [c for c in df_scored.columns if c not in id_cols and c not in TARGET_COLUMNS]
        X_batch = df_scored[feature_cols]

        probas = self.predict_proba(X_batch)
        preds = (probas >= thresh).astype(int)

        df_scored["Predicted_Probability"] = np.round(probas, 4)
        df_scored["Predicted_Outcome"] = preds
        df_scored["Risk_Tier"] = [classify_risk_tier(p)[0] for p in probas]

        return df_scored

    def save(self, filepath: str) -> str:
        """Serialize pipeline bundle to disk using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: str) -> "CervicalCancerPipeline":
        """Load serialized pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at '{filepath}'")
        pipeline = joblib.load(filepath)
        return pipeline


def train_and_save_default_model(
    output_path: Optional[str] = None,
    target_column: str = "Biopsy",
    model_name: str = "random_forest"
) -> Tuple[CervicalCancerPipeline, Dict[str, Any]]:
    """Convenience function to train on the UCI dataset and save the reference checkpoint."""
    if output_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_path = os.path.join(base_dir, "models", f"cervical_{model_name}_{target_column.lower()}.joblib")

    print(f"Loading UCI dataset for target '{target_column}'...")
    df = load_dataset()

    print(f"Instantiating pipeline (Model: {model_name}, Imbalance: SMOTE)...")
    pipeline = CervicalCancerPipeline(
        model_name=model_name,
        target_column=target_column,
        imbalance_strategy="smote",
        scaling_method="standard",
        missing_threshold=0.50,
        decision_threshold=0.50
    )

    print("Training and evaluating pipeline...")
    eval_results = pipeline.train_and_evaluate(df)

    print(f"Saving trained model pipeline to {output_path}...")
    pipeline.save(output_path)

    print(f"Model successfully saved! Test ROC-AUC: {pipeline.test_metrics_.get('roc_auc', 0):.3f}")
    return pipeline, eval_results
