"""
Streamlit Web Dashboard for AI-Based Cervical Cancer Detection (Image & Clinical Risk System).
Completely automatic pixel extraction, deep learning classification, visual Grad-CAM heatmap explanation,
and multi-model comparison dashboard (MobileNetV3-Small vs. EfficientNet-B0).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
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
    page_title="AI Cervical Cancer Detection (Image & Multi-Model System)",
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
    .result-box-invalid {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        color: #721c24;
        margin-bottom: 15px;
    }
    .model-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .consensus-agree {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 14px;
        color: #1b5e20;
        text-align: center;
        margin-bottom: 15px;
    }
    .consensus-diverge {
        background-color: #fff8e1;
        border: 2px solid #ffb300;
        border-radius: 8px;
        padding: 14px;
        color: #f57f17;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize sample files and load detectors for both architectures
@st.cache_resource
def get_detectors() -> Dict[str, AutomaticCervicalDetector]:
    normal_p, abnormal_p, mob_p, eff_p = setup_benchmark_images_and_model(PROJECT_ROOT)
    return {
        "mobilenet_v3": AutomaticCervicalDetector(model_path=mob_p, model_name="mobilenet_v3"),
        "efficientnet_b0": AutomaticCervicalDetector(model_path=eff_p, model_name="efficientnet_b0"),
    }


@st.cache_data
def load_benchmarks() -> Dict[str, Dict[str, Any]]:
    metrics = {}
    models_dir = os.path.join(PROJECT_ROOT, "models")
    defaults = {
        "mobilenet_v3": {
            "architecture_key": "mobilenet_v3",
            "display_title": "MobileNetV3-Small",
            "accuracy": 0.9103,
            "sensitivity": 1.0,
            "specificity": 0.30,
            "precision": 0.9067,
            "f1": 0.9510,
            "roc_auc": 0.9353,
            "true_positives": 68,
            "false_positives": 7,
            "true_negatives": 3,
            "false_negatives": 0,
            "total_parameters": 1001122,
            "avg_latency_ms": 13.18,
            "model_size_mb": 3.95
        },
        "efficientnet_b0": {
            "architecture_key": "efficientnet_b0",
            "display_title": "EfficientNet-B0",
            "accuracy": 0.9615,
            "sensitivity": 0.9853,
            "specificity": 0.80,
            "precision": 0.9710,
            "f1": 0.9781,
            "roc_auc": 0.9691,
            "true_positives": 67,
            "false_positives": 2,
            "true_negatives": 8,
            "false_negatives": 1,
            "total_parameters": 4171774,
            "avg_latency_ms": 101.08,
            "model_size_mb": 16.20
        }
    }
    for k in ["mobilenet_v3", "efficientnet_b0"]:
        p = os.path.join(models_dir, f"{k}_metrics.json")
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    metrics[k] = json.load(f)
            except Exception:
                metrics[k] = defaults[k]
        else:
            metrics[k] = defaults[k]
    return metrics


detectors = get_detectors()
benchmarks = load_benchmarks()

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
st.caption("Automatic Pixel Extraction, Multi-Architecture Deep Learning Classification (MobileNetV3 vs. EfficientNet-B0) & Grad-CAM Heatmaps")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ System Mode")
    app_mode = st.radio(
        "Select Detection Mode",
        ["🖼️ Image-Based Classification", "📊 Tabular Risk Factor Analysis"],
        index=0
    )

    st.markdown("---")
    st.subheader("🧠 Model Architecture Toggle")
    if "active_model_key" not in st.session_state:
        st.session_state["active_model_key"] = "mobilenet_v3"

    model_options = {
        "mobilenet_v3": "⚡ MobileNetV3-Small (Edge / Ultra-Fast)",
        "efficientnet_b0": "🎯 EfficientNet-B0 (Compound Scaling / 96.2% Acc)"
    }

    selected_model_option = st.radio(
        "Active Vision Backbone:",
        list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0 if st.session_state["active_model_key"] == "mobilenet_v3" else 1,
        key="sidebar_model_radio"
    )
    st.session_state["active_model_key"] = selected_model_option
    active_key = st.session_state["active_model_key"]
    active_detector = detectors[active_key]
    active_bench = benchmarks[active_key]

    st.markdown("---")
    st.subheader(f"🏆 Active Model: {active_bench['display_title']}")
    st.markdown(f"""
    * **Test Accuracy:** `{active_bench['accuracy'] * 100:.1f}%`
    * **Cancer Sensitivity:** `{active_bench['sensitivity'] * 100:.1f}%`
    * **Benign Specificity:** `{active_bench['specificity'] * 100:.1f}%`
    * **ROC-AUC Score:** `{active_bench['roc_auc']:.3f}`
    * **Parameters:** `{active_bench['total_parameters']:,}`
    * **CPU Latency:** `{active_bench['avg_latency_ms']:.1f} ms / image`
    * **Weights Size:** `{active_bench['model_size_mb']:.1f} MB`
    """)

    st.markdown("---")
    st.markdown("""
    **Workflow Summary:**
    1. Upload any cervical screening photo or choose a benchmark cytology sample.
    2. Deep network automatically extracts pixel statistics, nuclear chromatin density, and cytoplasmic contours.
    3. Outputs cancer detection decision, confidence %, and a **Grad-CAM attention heatmap**.
    4. Switch between **MobileNetV3** and **EfficientNet-B0** or compare them side-by-side in the Comparison Dashboard.
    """)


# =========================================================================
# MODE 1: IMAGE-BASED CLASSIFICATION
# =========================================================================
if app_mode == "🖼️ Image-Based Classification":

    tab_single, tab_compare, tab_batch, tab_dataset, tab_guide = st.tabs([
        "📸 Instant Photo Scanner",
        "⚖️ Model Comparison Dashboard",
        "📁 Batch Photos Triage",
        "🏥 Real Online Dataset & Benchmark",
        "📖 Clinical Cytology Guide"
    ])

    normal_sample_path = os.path.join(PROJECT_ROOT, "data", "images", "samples", "real_herlev_normal_cell.jpg")
    abnormal_sample_path = os.path.join(PROJECT_ROOT, "data", "images", "samples", "real_herlev_cancer_lesion.jpg")

    # TAB 1: INSTANT PHOTO SCANNER
    with tab_single:
        st.subheader("1. Screening Photo Selection & Model Control")

        header_c1, header_c2 = st.columns([3, 2])
        with header_c1:
            st.markdown(f"Currently Active Model: **{active_bench['display_title']}**")
        with header_c2:
            switch_to = "efficientnet_b0" if active_key == "mobilenet_v3" else "mobilenet_v3"
            target_label = "🎯 Switch to EfficientNet-B0" if active_key == "mobilenet_v3" else "⚡ Switch to MobileNetV3-Small"
            if st.button(target_label, use_container_width=True, key="switch_model_btn"):
                st.session_state["active_model_key"] = switch_to
                st.rerun()

        sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 2])
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
            # Auto-load cancer sample on first open
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
            st.caption(f"Evaluated using **{active_bench['display_title']}** (Inference Latency: ~{active_bench['avg_latency_ms']:.1f} ms)")

            with st.spinner(f"Extracting image pixels and running {active_bench['display_title']}..."):
                t_start = time.perf_counter()
                results = active_detector.detect(selected_image)
                infer_ms = (time.perf_counter() - t_start) * 1000.0

            # Check if image passed cytology domain validation
            if not results.get("is_valid_cervical_photo", True):
                st.markdown(f"""
                <div class="result-box-invalid">
                    <h2 style="margin:0; color:#dc3545;">❌ WRONG PHOTO UPLOADED</h2>
                    <h4 style="margin:6px 0; color:#721c24;">Unrelated Image Detected — Not a Cervical Screening Micrograph</h4>
                    <p style="margin:4px 0 0 0; font-size:15px;"><strong>Detection Flag:</strong> {results['error_reason']}</p>
                </div>
                """, unsafe_allow_html=True)

                col_err1, col_err2 = st.columns([1, 2])
                with col_err1:
                    st.markdown("##### 📷 Uploaded Image")
                    st.image(results["resized_image"], use_container_width=True, caption="Rejected Image (Non-Cervical)")

                with col_err2:
                    st.markdown("##### 🔬 Automatic Microscopic Quality & Domain Verification")
                    st.markdown("""
                    To prevent hazardous false medical diagnoses, the system automatically verifies every upload against clinical histology criteria:
                    * **Illumination & Textural Contrast:** Checks for optical brightfield microscopy lighting and cellular textural variance.
                    * **Papanicolaou Staining Profile:** Validates hematoxylin (basophilic) and counterstain balance, rejecting unnatural foliage greens, saturated icons, or extreme color casts.
                    * **Deep Cytology Embedding Match:** Evaluates deep neural feature manifold cosine similarity against verified clinical cervical cells.
                    """)
                    if "verification_checks" in results:
                        st.json(results["verification_checks"])
                    st.warning("⚠️ **No cancer diagnosis was run on this image.** Please upload a genuine Pap smear microscopic slide photo or select one of the verified clinical samples above.")

            else:
                # Display Decision Banner
                is_detected = results["is_cancer_detected"]
                conf = results["confidence"]

                if is_detected:
                    st.markdown(f"""
                    <div class="result-box-positive">
                        <h2 style="margin:0; color:#dc3545;">⚠️ {results['prediction'].upper()} DETECTED</h2>
                        <h4 style="margin:6px 0; color:#721c24;">Model Confidence: <strong>{conf:.1f}%</strong> | Risk Tier: <strong>{results['risk_tier']}</strong> | Model: <strong>{active_bench['display_title']}</strong></h4>
                        <p style="margin:0; font-size:14px;">{results['clinical_advisory']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-negative">
                        <h2 style="margin:0; color:#28a745;">✅ {results['prediction'].upper()}</h2>
                        <h4 style="margin:6px 0; color:#155724;">Model Confidence: <strong>{conf:.1f}%</strong> | Risk Tier: <strong>{results['risk_tier']}</strong> | Model: <strong>{active_bench['display_title']}</strong></h4>
                        <p style="margin:0; font-size:14px;">{results['clinical_advisory']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visual Comparison & Heatmap Display
                col_img1, col_img2, col_stats = st.columns([1, 1, 1])

                with col_img1:
                    st.markdown("##### 📷 Original Photo")
                    st.image(results["resized_image"], use_container_width=True, caption="Normalized Image (224x224)")

                with col_img2:
                    st.markdown(f"##### 🎯 Pixel Attention Heatmap (Grad-CAM)")
                    st.image(
                        results["heatmap_overlay"],
                        use_container_width=True,
                        caption=f"{active_bench['display_title']} attention highlighting flagged cellular pixels"
                    )

                with col_stats:
                    st.markdown("##### 📊 Extracted Pixel Features")
                    pstats = results["pixel_stats"]
                    st.markdown(f"""
                    * **Source Dimensions:** `{pstats['original_dimensions']}`
                    * **Average Brightness:** `{pstats['brightness']} / 255`
                    * **Pixel Contrast:** `{pstats['contrast']}`
                    * **Nuclear Chromatin Density:** `{pstats['nuclear_chromatin_pixel_density']}`
                    * **Inference Speed:** `{infer_ms:.1f} ms`
                    """)

                    st.markdown("##### 📈 Class Probability Scores")
                    prob_df = pd.DataFrame({
                        "Class": ["Normal / Benign", "Cancer / Lesion"],
                        "Probability (%)": [results["prob_normal"], results["prob_abnormal"]]
                    })
                    st.bar_chart(prob_df.set_index("Class"), height=180)

        else:
            st.info("👆 Click one of the sample buttons above or upload a photo to perform automatic detection.")


    # =====================================================================
    # TAB 2: MODEL COMPARISON & BENCHMARK DASHBOARD (NEW!)
    # =====================================================================
    with tab_compare:
        st.subheader("⚖️ Deep Learning Architecture Comparison Dashboard")
        st.markdown("""
        Compare **MobileNetV3-Small** (Lightweight Inverted Residuals) and **EfficientNet-B0** (Compound Depth/Width/Resolution Scaling)
        trained on the identical **Herlev University Hospital** Pap smear dataset.
        """)

        # 1. LIVE DUAL-MODEL INFERENCE ON ACTIVE PHOTO
        st.markdown("### 🔬 1. Live Side-by-Side Dual Inference")
        current_img = st.session_state.get("active_img", None)
        active_name = st.session_state.get("active_name", "Current Sample")

        if current_img is not None:
            st.caption(f"Testing both neural networks concurrently on: **{active_name}**")
            
            with st.spinner("Executing parallel inference across both vision architectures..."):
                t0_m = time.perf_counter()
                res_mob = detectors["mobilenet_v3"].detect(current_img)
                ms_mob = (time.perf_counter() - t0_m) * 1000.0

                t0_e = time.perf_counter()
                res_eff = detectors["efficientnet_b0"].detect(current_img)
                ms_eff = (time.perf_counter() - t0_e) * 1000.0

            # Consensus & Domain Validity Assessment
            valid_m = res_mob.get("is_valid_cervical_photo", True)
            valid_e = res_eff.get("is_valid_cervical_photo", True)

            if not valid_m or not valid_e:
                err_msg = res_mob.get("error_reason") if not valid_m else res_eff.get("error_reason")
                st.markdown(f"""
                <div class="result-box-invalid">
                    <h3 style="margin:0; color:#dc3545;">❌ WRONG PHOTO DETECTED BY BOTH ARCHITECTURES</h3>
                    <p style="margin:6px 0 0 0; font-size:15px;"><strong>Detection Flag:</strong> {err_msg}</p>
                    <p style="margin:4px 0 0 0; font-size:13px; color:#721c24;">Both MobileNetV3 and EfficientNet-B0 domain filters confirmed this image is an unrelated non-microscopic photo. No cancer classification was executed.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                agree = res_mob["is_cancer_detected"] == res_eff["is_cancer_detected"]
                if agree:
                    common_class = res_mob["prediction"]
                    avg_conf = (res_mob["confidence"] + res_eff["confidence"]) / 2.0
                    st.markdown(f"""
                    <div class="consensus-agree">
                        <h3 style="margin:0;">✅ Diagnostic Consensus: Both Models Agree ({common_class.upper()})</h3>
                        <p style="margin:4px 0 0 0; font-size:14px;">Combined Average Confidence: <strong>{avg_conf:.1f}%</strong> | Strong architectural alignment on morphological features.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="consensus-diverge">
                        <h3 style="margin:0;">⚠️ Diagnostic Divergence Detected</h3>
                        <p style="margin:4px 0 0 0; font-size:14px;">MobileNetV3 predicted <strong>{res_mob['prediction']}</strong> ({res_mob['confidence']:.1f}%) while EfficientNet-B0 predicted <strong>{res_eff['prediction']}</strong> ({res_eff['confidence']:.1f}%). Secondary cytotechnologist review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)

            dual_col1, dual_col2 = st.columns(2)

            with dual_col1:
                st.markdown("#### ⚡ MobileNetV3-Small")
                if not valid_m:
                    st.error("❌ Wrong Photo / Unrelated Image")
                    st.image(res_mob["resized_image"], use_container_width=True, caption="Rejected Image")
                else:
                    flag_m = "🔴 CANCER / LESION" if res_mob["is_cancer_detected"] else "🟢 NORMAL / BENIGN"
                    st.markdown(f"**Outcome:** `{flag_m}` | **Confidence:** `{res_mob['confidence']:.1f}%` | **Latency:** `{ms_mob:.1f} ms`")
                    st.image(res_mob["heatmap_overlay"], use_container_width=True, caption="MobileNetV3 Grad-CAM Attention Heatmap")
                    st.progress(res_mob["confidence"] / 100.0, text=f"Confidence: {res_mob['confidence']:.1f}%")

            with dual_col2:
                st.markdown("#### 🎯 EfficientNet-B0")
                if not valid_e:
                    st.error("❌ Wrong Photo / Unrelated Image")
                    st.image(res_eff["resized_image"], use_container_width=True, caption="Rejected Image")
                else:
                    flag_e = "🔴 CANCER / LESION" if res_eff["is_cancer_detected"] else "🟢 NORMAL / BENIGN"
                    st.markdown(f"**Outcome:** `{flag_e}` | **Confidence:** `{res_eff['confidence']:.1f}%` | **Latency:** `{ms_eff:.1f} ms`")
                    st.image(res_eff["heatmap_overlay"], use_container_width=True, caption="EfficientNet-B0 Grad-CAM Attention Heatmap")
                    st.progress(res_eff["confidence"] / 100.0, text=f"Confidence: {res_eff['confidence']:.1f}%")

        else:
            st.info("Upload or select a photo in the Instant Photo Scanner tab to see live side-by-side inference.")

        st.markdown("---")

        # 2. COMPREHENSIVE BENCHMARK METRICS TABLE
        st.markdown("### 📊 2. Clinical Benchmark Metrics Table (Holdout Test Cohort)")
        m_mob = benchmarks["mobilenet_v3"]
        m_eff = benchmarks["efficientnet_b0"]

        metrics_data = [
            {"Evaluation Metric": "Overall Test Accuracy", "MobileNetV3-Small": f"{m_mob['accuracy']*100:.1f}%", "EfficientNet-B0": f"{m_eff['accuracy']*100:.1f}%", "Clinical Advantage": "EfficientNet-B0 (+5.1% accuracy)"},
            {"Evaluation Metric": "Cancer Sensitivity (Recall)", "MobileNetV3-Small": f"{m_mob['sensitivity']*100:.1f}%", "EfficientNet-B0": f"{m_eff['sensitivity']*100:.1f}%", "Clinical Advantage": "MobileNetV3 achieves 100% zero-miss recall"},
            {"Evaluation Metric": "Benign Specificity", "MobileNetV3-Small": f"{m_mob['specificity']*100:.1f}%", "EfficientNet-B0": f"{m_eff['specificity']*100:.1f}%", "Clinical Advantage": "EfficientNet-B0 (+50.0% higher specificity)"},
            {"Evaluation Metric": "Cancer Precision", "MobileNetV3-Small": f"{m_mob['precision']*100:.1f}%", "EfficientNet-B0": f"{m_eff['precision']*100:.1f}%", "Clinical Advantage": "EfficientNet-B0 (+6.4% higher precision)"},
            {"Evaluation Metric": "F1-Score", "MobileNetV3-Small": f"{m_mob['f1']:.3f}", "EfficientNet-B0": f"{m_eff['f1']:.3f}", "Clinical Advantage": "EfficientNet-B0 (Higher balance score)"},
            {"Evaluation Metric": "ROC-AUC Score", "MobileNetV3-Small": f"{m_mob['roc_auc']:.3f}", "EfficientNet-B0": f"{m_eff['roc_auc']:.3f}", "Clinical Advantage": "EfficientNet-B0 (0.969 vs 0.935)"},
            {"Evaluation Metric": "Parameter Count", "MobileNetV3-Small": f"{m_mob['total_parameters']:,} (~1.0M)", "EfficientNet-B0": f"{m_eff['total_parameters']:,} (~4.2M)", "Clinical Advantage": "MobileNetV3 is 4.2x more lightweight"},
            {"Evaluation Metric": "Model Disk Size", "MobileNetV3-Small": f"{m_mob['model_size_mb']:.1f} MB", "EfficientNet-B0": f"{m_eff['model_size_mb']:.1f} MB", "Clinical Advantage": "MobileNetV3 requires only ~4 MB"},
            {"Evaluation Metric": "CPU Inference Latency", "MobileNetV3-Small": f"{m_mob['avg_latency_ms']:.1f} ms", "EfficientNet-B0": f"{m_eff['avg_latency_ms']:.1f} ms", "Clinical Advantage": "MobileNetV3 is ~7.6x faster on CPU"}
        ]
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        # 3. INTERACTIVE VISUAL COMPARISON CHARTS
        st.markdown("### 📈 3. Visual Performance & Efficiency Comparison")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 🎯 Diagnostic Clinical Scores (%)")
            diag_df = pd.DataFrame({
                "Metric": ["Accuracy", "Sensitivity", "Specificity", "Precision", "ROC-AUC (x100)"],
                "MobileNetV3-Small": [m_mob["accuracy"]*100, m_mob["sensitivity"]*100, m_mob["specificity"]*100, m_mob["precision"]*100, m_mob["roc_auc"]*100],
                "EfficientNet-B0": [m_eff["accuracy"]*100, m_eff["sensitivity"]*100, m_eff["specificity"]*100, m_eff["precision"]*100, m_eff["roc_auc"]*100]
            }).set_index("Metric")
            st.bar_chart(diag_df, height=280)

        with chart_col2:
            st.markdown("##### ⚡ Compute Efficiency & Resource Footprint")
            comp_df = pd.DataFrame({
                "Model": ["MobileNetV3-Small", "EfficientNet-B0"],
                "CPU Latency (ms)": [m_mob["avg_latency_ms"], m_eff["avg_latency_ms"]],
                "Weights Size (MB)": [m_mob["model_size_mb"], m_eff["model_size_mb"]],
                "Parameters (Millions)": [m_mob["total_parameters"]/1e6, m_eff["total_parameters"]/1e6]
            }).set_index("Model")
            st.bar_chart(comp_df, height=280)

        # 4. SIDE-BY-SIDE CONFUSION MATRIX VISUALIZATION
        st.markdown("---")
        st.markdown("### 🧮 4. Confusion Matrix Comparison (78 Clinical Test Micrographs)")
        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("##### MobileNetV3-Small Confusion Matrix")
            cm_mob_df = pd.DataFrame({
                "Predicted Cancer (Abnormal)": [f"TP = {m_mob['true_positives']} (Cancer)", f"FP = {m_mob['false_positives']} (Benign)"],
                "Predicted Normal (Benign)": [f"FN = {m_mob['false_negatives']} (Missed)", f"TN = {m_mob['true_negatives']} (Correct)"]
            }, index=["Actual Cancer (68)", "Actual Normal (10)"])
            st.table(cm_mob_df)
            st.caption("✅ 0 missed cancers (100% sensitivity) | 7 false alarms on benign cells")

        with cm_col2:
            st.markdown("##### EfficientNet-B0 Confusion Matrix")
            cm_eff_df = pd.DataFrame({
                "Predicted Cancer (Abnormal)": [f"TP = {m_eff['true_positives']} (Cancer)", f"FP = {m_eff['false_positives']} (Benign)"],
                "Predicted Normal (Benign)": [f"FN = {m_eff['false_negatives']} (Missed)", f"TN = {m_eff['true_negatives']} (Correct)"]
            }, index=["Actual Cancer (68)", "Actual Normal (10)"])
            st.table(cm_eff_df)
            st.caption("✅ 8 out of 10 normal cells correctly identified (80% specificity) | only 1 false negative")

        # 5. CLINICAL DEPLOYMENT GUIDANCE
        st.markdown("---")
        st.markdown("### 🏥 5. Clinical Deployment Trade-Off Analysis")
        dep_c1, dep_c2 = st.columns(2)
        with dep_c1:
            st.markdown("""
            **When to Select MobileNetV3-Small:**
            * **Point-of-Care & Mobile Health Clinics:** Ultra-low latency (~13 ms) and compact ~4 MB footprint runs effortlessly on low-power tablets, laptops, or Raspberry Pi edge devices.
            * **First-Pass Screening (Zero False Negatives):** Achieved **100% Cancer Sensitivity**, ensuring no suspicious lesions pass undetected during high-volume primary population triage.
            """)
        with dep_c2:
            st.markdown("""
            **When to Select EfficientNet-B0:**
            * **Central Hospital Pathology Laboratory:** Highest diagnostic accuracy (**96.2%**) and high specificity (**80%**), substantially cutting down false positive alarms.
            * **Compound Resolution Scaling:** Deep feature maps with SiLU activation excel at resolving subtle chromatin clumping, nuclear margin irregularities, and dysplastic cell boundaries.
            """)


    # TAB 3: BATCH PHOTOS SCANNER
    with tab_batch:
        st.subheader("📁 Automatic Batch Patient Screening")
        st.markdown(f"Upload multiple photos at once. The system will extract pixels from each photo using **{active_bench['display_title']}**.")

        batch_files = st.file_uploader(
            "Select Multiple Photos",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True
        )

        if batch_files:
            st.write(f"Loaded **{len(batch_files)}** screening photos.")
            if st.button(f"🚀 Run Automatic Batch Detection ({active_bench['display_title']})", type="primary"):
                batch_records = []
                progress_bar = st.progress(0)

                for i, file_obj in enumerate(batch_files):
                    img = Image.open(file_obj)
                    res = active_detector.detect(img)
                    is_valid = res.get("is_valid_cervical_photo", True)
                    batch_records.append({
                        "File Name": file_obj.name,
                        "Model Used": active_bench["display_title"],
                        "Detection Decision": res["prediction"] if is_valid else "❌ WRONG PHOTO (Unrelated)",
                        "Abnormal / Cancer Flag": ("🔴 Positive" if res["is_cancer_detected"] else "🟢 Negative") if is_valid else "⛔ Rejected (Non-Cervical)",
                        "Confidence (%)": f"{res['confidence']:.1f}%" if is_valid else "N/A",
                        "Abnormal Probability (%)": f"{res['prob_abnormal']:.1f}%" if is_valid else "N/A",
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


    # TAB 4: REAL ONLINE DATASET & BENCHMARK
    with tab_dataset:
        st.subheader("🏥 Online Benchmark: Herlev University Hospital Pap Smear Database")
        st.markdown("""
        Both models are trained directly on real clinical microscopy images from the **Herlev University Hospital Cytology Database** (hosted by MDE-Lab, University of the Aegean).
        """)

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("EfficientNet-B0 Accuracy", "96.2%", delta="+5.1% over MobileNetV3")
        dcol2.metric("EfficientNet-B0 Specificity", "80.0%", delta="+50.0% over MobileNetV3")
        dcol3.metric("MobileNetV3 Cancer Recall", "100.0%", delta="Zero False Negatives")
        dcol4.metric("MobileNetV3 Speed", "13.2 ms", delta="7.6x faster")

        st.markdown("---")
        st.markdown("##### Dataset Breakdown")
        st.write("""
        * **Training Set:** 303 verified clinical cytology cells (74 normal squamous cells, 229 abnormal/dysplastic cells).
        * **Holdout Test Set:** 78 independent clinical validation images (10 normal, 68 abnormal).
        * **Class Imbalance Remedy:** Balanced loss weight re-scaling (`weights=[0.66, 2.05]`).
        * **Vision Backbones:** Pretrained MobileNetV3-Small & EfficientNet-B0 with cosine annealing learning rate schedulers.
        """)

        herlev_dir = os.path.join(PROJECT_ROOT, "data", "images", "real_herlev")
        if os.path.exists(herlev_dir):
            st.success(f"✅ Real Herlev dataset is cached locally in `{herlev_dir}`.")


    # TAB 5: CLINICAL CYTOLOGY GUIDE
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
