"""
Data loading and dataset management module for Cervical Cancer Risk Prediction.
Handles downloading, caching, schema validation, and parsing of medical screening records.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

UCI_CERVICAL_URL = "https://archive.ics.uci.edu/static/public/383/cervical+cancer+risk+factors.zip"
TARGET_COLUMNS = ["Biopsy", "Hinselmann", "Schiller", "Citology"]

# Human-friendly descriptions and category mappings
FEATURE_METADATA: Dict[str, Dict[str, Union[str, float, int]]] = {
    "Age": {"type": "numeric", "desc": "Patient age in years", "min": 13, "max": 85, "default": 28},
    "Number of sexual partners": {"type": "numeric", "desc": "Lifetime sexual partners", "min": 1, "max": 30, "default": 2},
    "First sexual intercourse": {"type": "numeric", "desc": "Age at first intercourse", "min": 10, "max": 35, "default": 17},
    "Num of pregnancies": {"type": "numeric", "desc": "Total number of pregnancies", "min": 0, "max": 15, "default": 1},
    "Smokes": {"type": "binary", "desc": "Current or past smoker indicator", "min": 0, "max": 1, "default": 0},
    "Smokes (years)": {"type": "numeric", "desc": "Years of smoking history", "min": 0, "max": 40, "default": 0},
    "Smokes (packs/year)": {"type": "numeric", "desc": "Packs per year", "min": 0, "max": 40, "default": 0},
    "Hormonal Contraceptives": {"type": "binary", "desc": "Hormonal contraceptive usage indicator", "min": 0, "max": 1, "default": 1},
    "Hormonal Contraceptives (years)": {"type": "numeric", "desc": "Years of hormonal contraceptive use", "min": 0, "max": 30, "default": 2},
    "IUD": {"type": "binary", "desc": "Intrauterine device usage indicator", "min": 0, "max": 1, "default": 0},
    "IUD (years)": {"type": "numeric", "desc": "Years of IUD use", "min": 0, "max": 20, "default": 0},
    "STDs": {"type": "binary", "desc": "History of any sexually transmitted diseases", "min": 0, "max": 1, "default": 0},
    "STDs (number)": {"type": "numeric", "desc": "Total number of diagnosed STDs", "min": 0, "max": 6, "default": 0},
    "STDs:condylomatosis": {"type": "binary", "desc": "History of Condylomatosis (genital warts)", "min": 0, "max": 1, "default": 0},
    "STDs:cervical condylomatosis": {"type": "binary", "desc": "History of Cervical condylomatosis", "min": 0, "max": 1, "default": 0},
    "STDs:vaginal condylomatosis": {"type": "binary", "desc": "History of Vaginal condylomatosis", "min": 0, "max": 1, "default": 0},
    "STDs:vulvo-perineal condylomatosis": {"type": "binary", "desc": "History of Vulvo-perineal condylomatosis", "min": 0, "max": 1, "default": 0},
    "STDs:syphilis": {"type": "binary", "desc": "History of Syphilis", "min": 0, "max": 1, "default": 0},
    "STDs:pelvic inflammatory disease": {"type": "binary", "desc": "History of Pelvic Inflammatory Disease (PID)", "min": 0, "max": 1, "default": 0},
    "STDs:genital herpes": {"type": "binary", "desc": "History of Genital Herpes", "min": 0, "max": 1, "default": 0},
    "STDs:molluscum contagiosum": {"type": "binary", "desc": "History of Molluscum Contagiosum", "min": 0, "max": 1, "default": 0},
    "STDs:AIDS": {"type": "binary", "desc": "History of AIDS", "min": 0, "max": 1, "default": 0},
    "STDs:HIV": {"type": "binary", "desc": "HIV Positive Status", "min": 0, "max": 1, "default": 0},
    "STDs:Hepatitis B": {"type": "binary", "desc": "Hepatitis B diagnosis", "min": 0, "max": 1, "default": 0},
    "STDs:HPV": {"type": "binary", "desc": "HPV (Human Papillomavirus) diagnosis", "min": 0, "max": 1, "default": 0},
    "STDs: Number of diagnosis": {"type": "numeric", "desc": "Number of STD diagnoses", "min": 0, "max": 4, "default": 0},
    "STDs: Time since first diagnosis": {"type": "numeric", "desc": "Years since first STD diagnosis", "min": 0, "max": 25, "default": 0},
    "STDs: Time since last diagnosis": {"type": "numeric", "desc": "Years since last STD diagnosis", "min": 0, "max": 25, "default": 0},
    "Dx:Cancer": {"type": "binary", "desc": "Prior medical diagnosis of Cancer", "min": 0, "max": 1, "default": 0},
    "Dx:CIN": {"type": "binary", "desc": "Prior diagnosis of Cervical Intraepithelial Neoplasia", "min": 0, "max": 1, "default": 0},
    "Dx:HPV": {"type": "binary", "desc": "Prior clinical diagnosis of HPV", "min": 0, "max": 1, "default": 0},
    "Dx": {"type": "binary", "desc": "General clinical risk diagnosis indicator", "min": 0, "max": 1, "default": 0},
}


def get_default_data_dir() -> str:
    """Return the absolute path to the local data directory."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "data")


def load_dataset(
    file_source: Optional[Union[str, io.BytesIO]] = None,
    data_dir: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load dataset from a file path, file-like object, or cached UCI repository file.
    Replaces '?' strings with NaN and casts columns to numeric.
    """
    if file_source is not None:
        # User supplied a path or file-like object (e.g. from Streamlit upload)
        df = pd.read_csv(file_source, na_values=["?", "NA", "nan", "NaN", ""])
    else:
        # Default UCI dataset path
        if data_dir is None:
            data_dir = get_default_data_dir()
        raw_csv_path = os.path.join(data_dir, "raw", "risk_factors_cervical_cancer.csv")

        if force_download or not os.path.exists(raw_csv_path):
            os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
            print(f"Fetching dataset from {UCI_CERVICAL_URL}...")
            resp = urllib.request.urlopen(UCI_CERVICAL_URL, timeout=30)
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            z.extract("risk_factors_cervical_cancer.csv", os.path.join(data_dir, "raw"))

        df = pd.read_csv(raw_csv_path, na_values=["?", "NA", "nan", "NaN", ""])

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = "Biopsy",
    include_other_screenings_as_features: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into feature matrix X and target vector y.
    By default, other screening target columns (Biopsy, Hinselmann, Schiller, Citology)
    are excluded from X to avoid circular target leakage during risk assessment.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(df.columns)}")

    # Target vector
    y = df[target_column].copy()

    # Drop target
    cols_to_drop = [target_column]
    if not include_other_screenings_as_features:
        for t in TARGET_COLUMNS:
            if t in df.columns and t not in cols_to_drop:
                cols_to_drop.append(t)

    X = df.drop(columns=cols_to_drop).copy()
    return X, y


def generate_sample_patients_csv(output_path: Optional[str] = None) -> str:
    """
    Generate a curated sample CSV of patient records representing various clinical risk profiles
    (low risk, moderate risk, high risk) for demonstration and batch testing.
    """
    if output_path is None:
        data_dir = get_default_data_dir()
        output_path = os.path.join(data_dir, "sample", "sample_patient_records.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample_records = [
        # Patient 1: Young, low risk, non-smoker, no STDs
        {
            "Patient_ID": "PT-001",
            "Age": 22, "Number of sexual partners": 1, "First sexual intercourse": 19,
            "Num of pregnancies": 0, "Smokes": 0, "Smokes (years)": 0, "Smokes (packs/year)": 0,
            "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 1.0,
            "IUD": 0, "IUD (years)": 0, "STDs": 0, "STDs (number)": 0,
            "STDs:condylomatosis": 0, "STDs:cervical condylomatosis": 0, "STDs:vaginal condylomatosis": 0,
            "STDs:vulvo-perineal condylomatosis": 0, "STDs:syphilis": 0, "STDs:pelvic inflammatory disease": 0,
            "STDs:genital herpes": 0, "STDs:molluscum contagiosum": 0, "STDs:AIDS": 0, "STDs:HIV": 0,
            "STDs:Hepatitis B": 0, "STDs:HPV": 0, "STDs: Number of diagnosis": 0,
            "Dx:Cancer": 0, "Dx:CIN": 0, "Dx:HPV": 0, "Dx": 0
        },
        # Patient 2: Moderate risk - older, multiple pregnancies, long hormonal contraceptive use
        {
            "Patient_ID": "PT-002",
            "Age": 38, "Number of sexual partners": 3, "First sexual intercourse": 16,
            "Num of pregnancies": 3, "Smokes": 0, "Smokes (years)": 0, "Smokes (packs/year)": 0,
            "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 10.0,
            "IUD": 0, "IUD (years)": 0, "STDs": 0, "STDs (number)": 0,
            "STDs:condylomatosis": 0, "STDs:cervical condylomatosis": 0, "STDs:vaginal condylomatosis": 0,
            "STDs:vulvo-perineal condylomatosis": 0, "STDs:syphilis": 0, "STDs:pelvic inflammatory disease": 0,
            "STDs:genital herpes": 0, "STDs:molluscum contagiosum": 0, "STDs:AIDS": 0, "STDs:HIV": 0,
            "STDs:Hepatitis B": 0, "STDs:HPV": 0, "STDs: Number of diagnosis": 0,
            "Dx:Cancer": 0, "Dx:CIN": 0, "Dx:HPV": 0, "Dx": 0
        },
        # Patient 3: High risk - heavy smoking, long oral contraceptive, multiple STDs, HPV history
        {
            "Patient_ID": "PT-003",
            "Age": 45, "Number of sexual partners": 6, "First sexual intercourse": 14,
            "Num of pregnancies": 4, "Smokes": 1, "Smokes (years)": 20.0, "Smokes (packs/year)": 15.0,
            "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 12.0,
            "IUD": 1, "IUD (years)": 5.0, "STDs": 1, "STDs (number)": 2,
            "STDs:condylomatosis": 1, "STDs:cervical condylomatosis": 0, "STDs:vaginal condylomatosis": 0,
            "STDs:vulvo-perineal condylomatosis": 1, "STDs:syphilis": 0, "STDs:pelvic inflammatory disease": 0,
            "STDs:genital herpes": 0, "STDs:molluscum contagiosum": 0, "STDs:AIDS": 0, "STDs:HIV": 0,
            "STDs:Hepatitis B": 0, "STDs:HPV": 1, "STDs: Number of diagnosis": 1,
            "Dx:Cancer": 1, "Dx:CIN": 0, "Dx:HPV": 1, "Dx": 1
        },
        # Patient 4: HIV positive, moderate smoking, high partner count
        {
            "Patient_ID": "PT-004",
            "Age": 32, "Number of sexual partners": 5, "First sexual intercourse": 15,
            "Num of pregnancies": 2, "Smokes": 1, "Smokes (years)": 8.0, "Smokes (packs/year)": 4.0,
            "Hormonal Contraceptives": 0, "Hormonal Contraceptives (years)": 0,
            "IUD": 0, "IUD (years)": 0, "STDs": 1, "STDs (number)": 1,
            "STDs:condylomatosis": 0, "STDs:cervical condylomatosis": 0, "STDs:vaginal condylomatosis": 0,
            "STDs:vulvo-perineal condylomatosis": 0, "STDs:syphilis": 0, "STDs:pelvic inflammatory disease": 0,
            "STDs:genital herpes": 0, "STDs:molluscum contagiosum": 0, "STDs:AIDS": 0, "STDs:HIV": 1,
            "STDs:Hepatitis B": 0, "STDs:HPV": 0, "STDs: Number of diagnosis": 1,
            "Dx:Cancer": 0, "Dx:CIN": 1, "Dx:HPV": 0, "Dx": 1
        },
        # Patient 5: Normal average screening visit
        {
            "Patient_ID": "PT-005",
            "Age": 27, "Number of sexual partners": 2, "First sexual intercourse": 18,
            "Num of pregnancies": 1, "Smokes": 0, "Smokes (years)": 0, "Smokes (packs/year)": 0,
            "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 3.0,
            "IUD": 0, "IUD (years)": 0, "STDs": 0, "STDs (number)": 0,
            "STDs:condylomatosis": 0, "STDs:cervical condylomatosis": 0, "STDs:vaginal condylomatosis": 0,
            "STDs:vulvo-perineal condylomatosis": 0, "STDs:syphilis": 0, "STDs:pelvic inflammatory disease": 0,
            "STDs:genital herpes": 0, "STDs:molluscum contagiosum": 0, "STDs:AIDS": 0, "STDs:HIV": 0,
            "STDs:Hepatitis B": 0, "STDs:HPV": 0, "STDs: Number of diagnosis": 0,
            "Dx:Cancer": 0, "Dx:CIN": 0, "Dx:HPV": 0, "Dx": 0
        }
    ]

    sample_df = pd.DataFrame(sample_records)
    sample_df.to_csv(output_path, index=False)
    return output_path
