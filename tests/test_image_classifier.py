"""
Unit tests for the automatic image-based cervical cancer detection module.
"""

import os
from PIL import Image
import pytest
import torch
from src.image_classifier import (
    AutomaticCervicalDetector,
    CervicalEfficientNetB0,
    CervicalTransferCNN,
    auto_preprocess_image,
    generate_gradcam_overlay,
    setup_benchmark_images_and_model,
)


@pytest.fixture
def sample_test_images(tmp_path):
    """
    Creates minimal synthetic cervical-look-alike images in a temp directory.
    Works cross-platform (Linux CI, macOS, Windows) — no absolute local paths.
    """
    # Pinkish-purple tones resembling a stained normal cervical cell
    normal_img = Image.new("RGB", (128, 128), color=(210, 180, 200))
    normal_p = str(tmp_path / "normal_cell.jpg")
    normal_img.save(normal_p, "JPEG", quality=90)

    # Darker, higher-density look resembling an abnormal/cancer cell
    abnormal_img = Image.new("RGB", (128, 128), color=(140, 80, 120))
    abnormal_p = str(tmp_path / "abnormal_cell.jpg")
    abnormal_img.save(abnormal_p, "JPEG", quality=90)

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


def test_efficientnet_b0_detection_and_gradcam(sample_test_images):
    _, abnormal_p = sample_test_images
    detector = AutomaticCervicalDetector(model_name="efficientnet_b0")
    assert detector.model_name == "efficientnet_b0"
    assert isinstance(detector.model, CervicalEfficientNetB0)

    info = detector.get_model_info()
    assert info["total_parameters"] > 3_000_000
    assert "EfficientNet-B0" in info["display_title"]

    res = detector.detect(abnormal_p)
    assert "prediction" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 100.0
    assert isinstance(res["heatmap_overlay"], Image.Image)
    assert res["heatmap_overlay"].size == (224, 224)


def test_model_switching(sample_test_images):
    normal_p, _ = sample_test_images
    detector = AutomaticCervicalDetector(model_name="mobilenet_v3")
    assert detector.model_name == "mobilenet_v3"
    assert isinstance(detector.model, CervicalTransferCNN)

    detector.switch_model("efficientnet_b0")
    assert detector.model_name == "efficientnet_b0"
    assert isinstance(detector.model, CervicalEfficientNetB0)

    res = detector.detect(normal_p)
    assert "prediction" in res
    assert res["model_name"] == "efficientnet_b0"


def test_save_and_load_weights(tmp_path):
    model_p = str(tmp_path / "test_model.pth")
    detector = AutomaticCervicalDetector()
    detector.save_model(model_p)
    assert os.path.exists(model_p)

    new_detector = AutomaticCervicalDetector(model_path=model_p)
    assert new_detector.is_loaded is True


def test_wrong_photo_detection_rejection(tmp_path):
    """Verify that non-cervical / unrelated images are automatically rejected as WRONG PHOTO."""
    detector = AutomaticCervicalDetector(model_name="mobilenet_v3")

    # 1. Severely dark photo (e.g. night scenery / black object)
    dark_p = str(tmp_path / "dark_photo.jpg")
    Image.new("RGB", (128, 128), color=(15, 20, 25)).save(dark_p)
    res_dark = detector.detect(dark_p)
    assert res_dark["is_valid_cervical_photo"] is False
    assert "Wrong Photo" in res_dark["prediction"]
    assert res_dark["confidence"] == 0.0
    assert "error_reason" in res_dark

    # 2. Outdoor landscape / green foliage
    green_p = str(tmp_path / "green_landscape.jpg")
    Image.new("RGB", (128, 128), color=(20, 160, 30)).save(green_p)
    res_green = detector.detect(green_p)
    assert res_green["is_valid_cervical_photo"] is False
    assert "Wrong Photo" in res_green["prediction"]

    # 3. Blank white screen
    blank_p = str(tmp_path / "blank_canvas.jpg")
    Image.new("RGB", (128, 128), color=(255, 255, 255)).save(blank_p)
    res_blank = detector.detect(blank_p)
    assert res_blank["is_valid_cervical_photo"] is False
    assert "Wrong Photo" in res_blank["prediction"]


def test_real_clinical_samples_pass_validation():
    """Verify that genuine cervical Pap smear samples pass domain verification."""
    detector = AutomaticCervicalDetector(model_name="mobilenet_v3")
    sample_p = "data/images/samples/real_herlev_cancer_lesion.jpg"
    if os.path.exists(sample_p):
        res = detector.detect(sample_p)
        assert res["is_valid_cervical_photo"] is True
        assert res["confidence"] > 0.0
        assert "prediction" in res
