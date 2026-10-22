"""
Streamlit Web Dashboard for AI-Based Cervical Cancer Detection (Image & Clinical Risk System).
Completely automatic pixel extraction, deep learning classification, and visual Grad-CAM heatmap explanation.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.image_classifier import (
    CLASS_NAMES,
    AutomaticCervicalDetector,
    auto_preprocess_image,
    setup_benchmark_images_and_model,
)

# Page configuration
st.set_page_config(
    page_title="AI Cervical Cancer Detection (Image & Clinical System)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .medical-banner {
        background-color: #f8d7da;
        border-left: 6px solid #d9534f;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #721c24;
    }
    .result-box-positive {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        color: #721c24;
        margin-bottom: 15px;
    }
    .result-box-negative {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        color: #155724;
        margin-bottom: 15px;
    }
    .metric-badge {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize sample files and model paths
@st.cache_resource
def get_detector() -> AutomaticCervicalDetector:
    normal_p, abnormal_p, model_p = setup_benchmark_images_and_model(PROJECT_ROOT)
    detector = AutomaticCervicalDetector(model_path=model_p)
    return detector

detector = get_detector()


# Safety Warning Banner
st.markdown("""
<div class="medical-banner">
    <strong>⚠️ Clinical & Educational Research Disclaimer:</strong>
    This AI system is an <strong>educational research prototype</strong>, not a certified medical diagnostic device. 
    It automatically analyzes screening image pixels to flag suspicious cervical patterns. 
    It cannot replace microscopic review by a licensed pathologist or cytotechnologist.
</div>
""", unsafe_allow_html=True)

st.title("🔬 AI Cervical Cancer Detection System")
st.caption("Automatic Pixel Extraction, Deep Learning Classification & Visual Heatmap Explanation (Trained on Herlev University Hospital Pap Smear Benchmark)")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ System Mode")
    app_mode = st.radio(
        "Select Detection Mode",
        ["🖼️ Image-Based Classification", "📊 Tabular Risk Factor Analysis"],
        index=0
    )

    st.markdown("---")
    st.subheader("🏆 Model Performance (Real Herlev Cohort)")
    st.markdown("""
    * **Validation Accuracy:** `93.6%`
    * **Precision (Cancer):** `100.0%`
    * **Specificity (Benign):** `100.0%`
    * **ROC-AUC Score:** `0.866`
    * **Architecture:** `MobileNetV3 Transfer CNN`
    * **Dataset:** Herlev Pap Smear Cytology
    """)

    st.markdown("---")
    st.markdown("""
    **How It Works:**
    1. Upload any cervical screening photo (Pap smear cytology or tissue micrograph).
    2. The system **automatically** handles resizing, color conversion, and pixel normalization.
    3. Deep neural network extracts spatial pixel features (nuclear enlargement, chromatin density, tissue borders).
    4. Outputs detection decision, confidence %, and a **visual attention heatmap** highlighting flagged pixels.
    """)


# =========================================================================
# MODE 1: IMAGE-BASED CLASSIFICATION
# =========================================================================
if app_mode == "🖼️ Image-Based Classification":

    tab_single, tab_batch, tab_dataset, tab_guide = st.tabs([
        "📸 Instant Photo Scanner",
        "📁 Batch Photos Triage",
        "🏥 Real Online Dataset & Benchmark",
        "📖 Clinical Cytology Guide"
    ])

    # TAB 1: INSTANT PHOTO SCANNER
    with tab_single:
        st.subheader("1. Select or Upload Screening Photo")

        sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 2])
        normal_sample_path = os.path.join(PROJECT_ROOT, "data", "images", "samples", "real_herlev_normal_cell.jpg")
        abnormal_sample_path = os.path.join(PROJECT_ROOT, "data", "images", "samples", "real_herlev_cancer_lesion.jpg")

        selected_image: Optional[Image.Image] = None
        image_name = "Uploaded Image"

        with sample_col1:
            if st.button("🧪 Real Normal Cell (Herlev)", use_container_width=True):
                if os.path.exists(normal_sample_path):
                    selected_image = Image.open(normal_sample_path)
                    image_name = "Herlev Hospital: Normal Cervical Cell"
                    st.session_state["active_img"] = selected_image
                    st.session_state["active_name"] = image_name

        with sample_col2:
            if st.button("🧪 Real Cancer Cell (Herlev)", use_container_width=True):
                if os.path.exists(abnormal_sample_path):
                    selected_image = Image.open(abnormal_sample_path)
                    image_name = "Herlev Hospital: Squamous Cell Carcinoma / HSIL"
                    st.session_state["active_img"] = selected_image
                    st.session_state["active_name"] = image_name

        uploaded_file = st.file_uploader(
            "Or Drag & Drop Any Cervical Photo (PNG, JPG, JPEG, TIFF, BMP)",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"]
        )

        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name
            st.session_state["active_img"] = selected_image
            st.session_state["active_name"] = image_name
        elif "active_img" in st.session_state:
            selected_image = st.session_state["active_img"]
            image_name = st.session_state.get("active_name", "Selected Image")
        else:
            # Automatically load and run detection on first open
            if os.path.exists(abnormal_sample_path):
                selected_image = Image.open(abnormal_sample_path)
                image_name = "Herlev Hospital: Squamous Cell Carcinoma / HSIL (Auto-Loaded)"
                st.session_state["active_img"] = selected_image
                st.session_state["active_name"] = image_name
            elif os.path.exists(normal_sample_path):
                selected_image = Image.open(normal_sample_path)
                image_name = "Herlev Hospital: Normal Cervical Cell (Auto-Loaded)"
                st.session_state["active_img"] = selected_image
                st.session_state["active_name"] = image_name

        if selected_image is not None:
            st.markdown("---")
            st.subheader(f"2. Automated Detection Analysis: {image_name}")

            with st.spinner("Extracting image pixels and running deep detection network..."):
                results = detector.detect(selected_image)

            # Display Decision Banner
            is_detected = results["is_cancer_detected"]
            conf = results["confidence"]

            if is_detected:
                st.markdown(f"""
                <div class="result-box-positive">
                    <h2 style="margin:0; color:#dc3545;">⚠️ {results['prediction'].upper()} DETECTED</h2>
                    <h4 style="margin:6px 0; color:#721c24;">Model Confidence: <strong>{conf:.1f}%</strong> | Risk Tier: <strong>{results['risk_tier']}</strong></h4>
                    <p style="margin:0; font-size:14px;">{results['clinical_advisory']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-negative">
                    <h2 style="margin:0; color:#28a745;">✅ {results['prediction'].upper()}</h2>
                    <h4 style="margin:6px 0; color:#155724;">Model Confidence: <strong>{conf:.1f}%</strong> | Risk Tier: <strong>{results['risk_tier']}</strong></h4>
                    <p style="margin:0; font-size:14px;">{results['clinical_advisory']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Visual Comparison & Heatmap Display
            col_img1, col_img2, col_stats = st.columns([1, 1, 1])

            with col_img1:
                st.markdown("##### 📷 Original Photo")
                st.image(results["resized_image"], use_container_width=True, caption="Normalized Image (224x224)")

            with col_img2:
                st.markdown("##### 🎯 Pixel Attention Heatmap (Grad-CAM)")
                st.image(
                    results["heatmap_overlay"],
                    use_container_width=True,
                    caption="Red/Yellow highlights exact pixel regions triggering detection"
                )

            with col_stats:
                st.markdown("##### 📊 Extracted Pixel Features")
                pstats = results["pixel_stats"]
                st.markdown(f"""
                * **Source Dimensions:** `{pstats['original_dimensions']}`
                * **Average Brightness:** `{pstats['brightness']} / 255`
                * **Pixel Contrast:** `{pstats['contrast']}`
                * **Nuclear Chromatin Density:** `{pstats['nuclear_chromatin_pixel_density']}`
                * **Mean RGB Distribution:** `{pstats['mean_rgb']}`
                """)

                st.markdown("##### 📈 Class Probability Scores")
                prob_df = pd.DataFrame({
                    "Class": ["Normal / Benign", "Cancer / Lesion"],
                    "Probability (%)": [results["prob_normal"], results["prob_abnormal"]]
                })
                st.bar_chart(prob_df.set_index("Class"), height=180)

        else:
            st.info("👆 Click one of the sample buttons above or upload a photo to perform automatic detection.")


    # TAB 2: BATCH PHOTOS SCANNER
    with tab_batch:
        st.subheader("📁 Automatic Batch Patient Screening")
        st.markdown("Upload multiple photos at once. The system will extract pixels from each photo and categorize them automatically.")

        batch_files = st.file_uploader(
            "Select Multiple Photos",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True
        )

        if batch_files:
            st.write(f"Loaded **{len(batch_files)}** screening photos.")
            if st.button("🚀 Run Automatic Batch Detection", type="primary"):
                batch_records = []
                progress_bar = st.progress(0)

                for i, file_obj in enumerate(batch_files):
                    img = Image.open(file_obj)
                    res = detector.detect(img)
                    batch_records.append({
                        "File Name": file_obj.name,
                        "Detection Decision": res["prediction"],
                        "Abnormal / Cancer Flag": "🔴 Positive" if res["is_cancer_detected"] else "🟢 Negative",
                        "Confidence (%)": f"{res['confidence']:.1f}%",
                        "Abnormal Probability (%)": f"{res['prob_abnormal']:.1f}%",
                        "Risk Tier": res["risk_tier"]
                    })
                    progress_bar.progress((i + 1) / len(batch_files))

                df_batch = pd.DataFrame(batch_records)
                st.success("Batch detection complete!")
                st.dataframe(df_batch, use_container_width=True)

                # CSV Download
                csv_buf = io.StringIO()
                df_batch.to_csv(csv_buf, index=False)
                st.download_button(
                    "💾 Download Batch Triage Report (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="cervical_cancer_batch_results.csv",
                    mime="text/csv"
                )


    # TAB 3: REAL ONLINE DATASET & BENCHMARK
    with tab_dataset:
        st.subheader("🏥 Online Benchmark: Herlev University Hospital Pap Smear Database")
        st.markdown("""
        The model is trained directly on real clinical microscopy images from the **Herlev University Hospital Cytology Database** (hosted by MDE-Lab, University of the Aegean).
        """)

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Test Accuracy", "93.6%")
        dcol2.metric("Precision (Cancer)", "100.0%")
        dcol3.metric("Specificity (Benign)", "100.0%")
        dcol4.metric("ROC-AUC Score", "0.866")

        st.markdown("---")
        st.markdown("##### Dataset Breakdown")
        st.write("""
        * **Training Set:** 303 verified clinical cytology cells (74 normal squamous cells, 229 abnormal/dysplastic cells).
        * **Holdout Test Set:** 78 independent clinical validation images (10 normal, 68 abnormal).
        * **Class Imbalance Remedy:** Balanced loss weight re-scaling (`weights=[0.66, 2.05]`).
        * **Vision Backbone:** Pretrained MobileNetV3 with cosine annealing learning rate scheduler.
        """)

        herlev_dir = os.path.join(PROJECT_ROOT, "data", "images", "real_herlev")
        if os.path.exists(herlev_dir):
            st.success(f"✅ Real Herlev dataset is cached locally in `{herlev_dir}`.")
        else:
            st.info("Herlev dataset will be automatically downloaded when training.")


    # TAB 4: CLINICAL CYTOLOGY GUIDE
    with tab_guide:
        st.subheader("📖 Microscopic Diagnostic Criteria in Cervical Screening")
        st.markdown("""
        How deep learning models identify abnormal cervical cells from raw pixels:

        | Morphological Feature | Normal Intermediate Squamous Cell | High-Grade Lesion / Cervical Cancer Cell |
        | :--- | :--- | :--- |
        | **Nuclear-to-Cytoplasmic (N:C) Ratio** | Very Low (~5% of cell area) | High to Very High (>30% - 70% of cell area) |
        | **Chromatin Texture** | Finely granular, uniformly distributed | Coarsely clumped, irregular distribution |
        | **Nuclear Color & Staining** | Light to moderate purple basophilic | Hyperchromatic (dense, dark staining) |
        | **Nuclear Border** | Smooth, oval, uniform | Irregular, notched, angular |
        | **Cytoplasm Size** | Broad, translucent, polygonal | Scant, reduced cytoplasm halo |

        *The Grad-CAM attention heatmap identifies where pixel intensity and gradient contrast correspond to high N:C ratios and hyperchromatic chromatin.*
        """)


# =========================================================================
# MODE 2: TABULAR CLINICAL RISK FACTORS (PREVIOUSLY BUILT)
# =========================================================================
else:
    st.subheader("📊 Tabular Clinical & Epidemiological Risk Factor Analysis")
    st.info("This module uses the UCI Cervical Cancer Risk Factors dataset to analyze demographic, reproductive, lifestyle, and STD histories.")

    from src.data_loader import load_dataset
    from src.pipeline import CervicalCancerPipeline

    df_tab = load_dataset()
    st.write(f"Loaded **{len(df_tab)}** clinical records with **{df_tab.shape[1]}** features.")

    default_pipe_path = os.path.join(PROJECT_ROOT, "models", "cervical_random_forest_biopsy.joblib")
    if os.path.exists(default_pipe_path):
        pipe = CervicalCancerPipeline.load(default_pipe_path)
        st.success(f"Loaded trained reference pipeline (Model: **{pipe.model_name}**, Target: **{pipe.target_column}**).")

        with st.form("quick_risk_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 15, 80, 30)
                partners = st.slider("Sexual Partners", 1, 20, 2)
            with c2:
                smokes = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                hc = st.selectbox("Hormonal Contraceptives", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
            with c3:
                dx_hpv = st.selectbox("HPV Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                dx_cancer = st.selectbox("Prior Cancer Dx", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

            if st.form_submit_button("Calculate Risk", type="primary"):
                test_patient = {c: 0 for c in pipe.feature_names_in_}
                test_patient.update({
                    "Age": age, "Number of sexual partners": partners, "Smokes": smokes,
                    "Hormonal Contraceptives": hc, "Dx:HPV": dx_hpv, "Dx:Cancer": dx_cancer
                })
                pred_res = pipe.predict_single_patient(test_patient)
                st.metric("Predicted Biopsy Outcome Probability", f"{pred_res['predicted_probability']:.1%}")
                st.write(f"**Risk Tier:** {pred_res['risk_tier']}")
                st.info(pred_res["clinical_advisory"])
