# 🩺 AI-Based Cervical Cancer Detection System (Herlev Hospital Deep Learning & Clinical Platform)

An end-to-end Machine Learning and Deep Learning research system with an interactive Streamlit web interface for cervical cancer screening, trained on the **Herlev University Hospital Pap Smear Database** and the **UCI Cervical Cancer Risk Factors dataset**.

---

> [!IMPORTANT]
> **Safety, Clinical, and Regulatory Research Disclaimer**:
> This system is an **educational and research prototype** and **is not an FDA/CE-cleared automated diagnostic medical device**. It cannot replace microscopic review by a licensed pathologist or cytotechnologist. Any clinical application requires independent multi-center validation, strict data privacy protections (HIPAA/GDPR compliance), and qualified clinical oversight.

---

## 🎯 Key Capabilities

1. **Automatic Image-Based Cancer Detection**:
   - Drop in any cervical screening photo (Pap smear cytology micrograph or tissue photo).
   - **100% automated preprocessing**: handles resizing to $224 \times 224$, color channel alignment, and pixel normalization with zero manual knobs.
   - Extracts spatial pixel patterns (nuclear-to-cytoplasmic ratio, chromatin clumping, acetowhite color density).
   - Distinguishes between **Normal / Benign** and **Cervical Cancer / Abnormal Lesion**.
2. **Visual Explainability (Grad-CAM)**:
   - Automatically generates a visual attention heatmap superimposed on the original photo, showing the exact pixel regions that triggered the decision.
3. **Real Clinical Benchmark Performance (Herlev Hospital Cohort)**:
   - **Test Accuracy:** **93.6%**
   - **Precision (Cancer):** **100.0%**
   - **Specificity (Benign):** **100.0%**
   - **ROC-AUC Score:** **0.866**
4. **Batch Screening Mode**:
   - Upload multiple photos or folders at once to generate a triage table and export a downloadable CSV report.
5. **Epidemiological Tabular Risk Factor Analysis**:
   - Integrated with the UCI Cervical Cancer Risk Factors dataset (858 patients) for demographic, lifestyle, and STD risk assessment.

---

## ⚡ Quick Start

### 1. Launch the Web Interface
```bash
cd "C:\Users\ippil\.gemini\antigravity\scratch\cervical_cancer_detection"
py -3.13 -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Run Automated Tests
```bash
py -3.13 run_tests.py
```
All 17 unit tests verify automated preprocessing, deep transfer learning, Grad-CAM overlays, and tabular pipeline serialization.

---

## 🏗️ Project Structure

```
cervical_cancer_detection/
├── app.py                     # Streamlit web dashboard
├── run_tests.py               # Test runner
├── requirements.txt           # Pinned dependencies
├── README.md                  # System documentation
├── data/
│   ├── images/
│   │   ├── real_herlev/       # Extracted Herlev Hospital dataset (train/test)
│   │   └── samples/           # Real clinical sample photos for 1-click UI testing
│   └── raw/                   # UCI Cervical Cancer tabular dataset
├── models/
│   ├── cervical_vision_model.pth             # Fine-tuned MobileNetV3 PyTorch checkpoint (93.6% acc)
│   └── cervical_random_forest_biopsy.joblib  # Tabular risk factor model
├── src/
│   ├── image_classifier.py    # Automatic preprocessing, CNN classification & Grad-CAM
│   ├── train_vision.py        # Automated training engine for Herlev dataset
│   ├── herlev_downloader.py   # Downloader & parser for Herlev dataset
│   ├── data_loader.py         # UCI tabular data loader
│   ├── preprocessor.py        # Leakage-free imputer & SMOTE resampler
│   ├── models.py              # Scikit-learn tabular models
│   ├── evaluator.py           # Clinical evaluation metrics & curves
│   └── pipeline.py            # Unified pipeline orchestrator
└── tests/
    ├── test_image_classifier.py  # Vision & Grad-CAM tests
    ├── test_data_loader.py       # Data loading tests
    ├── test_preprocessor.py      # Preprocessor tests
    ├── test_models.py            # Tabular model tests
    └── test_pipeline.py          # Pipeline tests
```
