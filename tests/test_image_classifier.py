"""
Unit tests for the automatic image-based cervical cancer detection module.
"""

import os
from PIL import Image
import pytest
import torch
from src.image_classifier import (
    AutomaticCervicalDetector,
    CervicalTransferCNN,
    auto_preprocess_image,
    generate_gradcam_overlay,
    setup_benchmark_images_and_model,
)


@pytest.fixture
def sample_test_images(tmp_path):
    normal_p = os.path.abspath(r"C:\Users\ippil\.gemini\antigravity\scratch\cervical_cancer_detection\data\images\samples\real_herlev_normal_cell.jpg")
    abnormal_p = os.path.abspath(r"C:\Users\ippil\.gemini\antigravity\scratch\cervical_cancer_detection\data\images\samples\real_herlev_cancer_lesion.jpg")
    return normal_p, abnormal_p


def test_auto_preprocess_image(sample_test_images):
    normal_p, _ = sample_test_images
    tensor_img, resized_pil, stats = auto_preprocess_image(normal_p)

    assert tensor_img.shape == (1, 3, 224, 224)
    assert resized_pil.size == (224, 224)
    assert "mean_rgb" in stats
    assert "brightness" in stats
    assert "nuclear_chromatin_pixel_density" in stats


def test_automatic_detection(sample_test_images):
    _, abnormal_p = sample_test_images
    detector = AutomaticCervicalDetector()
    res = detector.detect(abnormal_p)

    assert "prediction" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 100.0
    assert "risk_tier" in res
    assert "heatmap_overlay" in res
    assert isinstance(res["heatmap_overlay"], Image.Image)


def test_gradcam_overlay_generation(sample_test_images):
    normal_p, _ = sample_test_images
    model = CervicalTransferCNN(pretrained=False)
    tensor_img, resized_pil, _ = auto_preprocess_image(normal_p)
    overlay = generate_gradcam_overlay(model, tensor_img, resized_pil, target_class_idx=0)

    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)


def test_save_and_load_weights(tmp_path):
    model_p = str(tmp_path / "test_model.pth")
    detector = AutomaticCervicalDetector()
    detector.save_model(model_p)
    assert os.path.exists(model_p)

    new_detector = AutomaticCervicalDetector(model_path=model_p)
    assert new_detector.is_loaded is True
