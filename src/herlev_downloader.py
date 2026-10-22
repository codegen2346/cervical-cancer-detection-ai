"""
Herlev Pap Smear Cytology Dataset Downloader and Extractor.
Downloads the real clinical cervical cytology dataset from MDE-Lab (Herlev University Hospital benchmark),
organizes real single-cell micrographs into 'normal' and 'abnormal' directories with train/test splits.
"""

from __future__ import annotations

import io
import os
import shutil
import urllib.request
import zipfile
import pandas as pd
from PIL import Image

HERLEV_SMEAR_URL = "https://mde-lab.aegean.gr/images/stories/docs/smear.zip"


def download_and_extract_herlev_dataset(
    target_dir: str,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Tuple[str, Dict[str, int]]:
    """
    Downloads smear.zip, reads labels from celler_150200-2-rev.txt,
    extracts images from celler_150200.zip, and structures them into:
      target_dir/train/normal
      target_dir/train/abnormal
      target_dir/test/normal
      target_dir/test/abnormal
    """
    os.makedirs(target_dir, exist_ok=True)
    cache_zip = os.path.join(target_dir, "smear.zip")

    if not os.path.exists(cache_zip):
        print(f"Downloading Herlev dataset from {HERLEV_SMEAR_URL} (~5.4 MB)...")
        req = urllib.request.Request(HERLEV_SMEAR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(cache_zip, "wb") as f:
            f.write(resp.read())
        print("Download complete!")

    with zipfile.ZipFile(cache_zip, "r") as z:
        # Read label file
        label_csv_bytes = z.read("smear/celler_150200-2-rev.txt")
        df_labels = pd.read_csv(io.StringIO(label_csv_bytes.decode("latin1")))

        # Read inner images zip
        inner_zip_bytes = io.BytesIO(z.read("smear/celler_150200.zip"))
        inner_z = zipfile.ZipFile(inner_zip_bytes)
        inner_files = {name.upper(): name for name in inner_z.namelist()}

    # Filter labeled records (0 = Normal, 1 = Abnormal)
    valid_df = df_labels[df_labels["RevDiag"].isin([0, 1])].copy()
    valid_df = valid_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Train / Test split
    test_size = int(len(valid_df) * test_ratio)
    test_df = valid_df.iloc[:test_size]
    train_df = valid_df.iloc[test_size:]

    splits = {"train": train_df, "test": test_df}
    counts = {"train_normal": 0, "train_abnormal": 0, "test_normal": 0, "test_abnormal": 0}

    # Extract images to split directories
    for split_name, split_data in splits.items():
        for _, row in split_data.iterrows():
            fname = str(row["FileName"]).strip()
            lookup_key = f"{fname}.JPG".upper()
            if lookup_key in inner_files:
                archive_name = inner_files[lookup_key]
                category = "abnormal" if row["RevDiag"] == 1 else "normal"
                dest_dir = os.path.join(target_dir, split_name, category)
                os.makedirs(dest_dir, exist_ok=True)

                dest_file = os.path.join(dest_dir, f"{fname}.jpg")
                img_data = inner_z.read(archive_name)

                # Ensure image is valid JPEG/RGB
                try:
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    pil_img.save(dest_file, "JPEG", quality=95)
                    counts[f"{split_name}_{category}"] += 1
                except Exception as e:
                    print(f"Warning: could not process {fname}: {e}")

    # Copy 2 representative real samples to samples directory for UI testing
    samples_dir = os.path.join(target_dir, "..", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Pick a clear normal and abnormal sample
    test_norm_dir = os.path.join(target_dir, "test", "normal")
    test_abnorm_dir = os.path.join(target_dir, "test", "abnormal")

    if os.path.exists(test_norm_dir) and os.listdir(test_norm_dir):
        src_norm = os.path.join(test_norm_dir, os.listdir(test_norm_dir)[0])
        shutil.copy(src_norm, os.path.join(samples_dir, "real_herlev_normal_cell.jpg"))

    if os.path.exists(test_abnorm_dir) and os.listdir(test_abnorm_dir):
        src_abnorm = os.path.join(test_abnorm_dir, os.listdir(test_abnorm_dir)[0])
        shutil.copy(src_abnorm, os.path.join(samples_dir, "real_herlev_cancer_lesion.jpg"))

    print(f"Extraction summary: {counts}")
    return target_dir, counts


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images", "real_herlev"))
    download_and_extract_herlev_dataset(base_dir)
