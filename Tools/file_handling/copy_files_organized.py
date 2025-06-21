import os
import shutil
from pathlib import Path

# Source and destination directories
SRC_DIR = r"D:\EMS_Data\EgoExoEMS_CVPR2025\Dataset\Final\wa1\cardiac_scenario"
DEST_DIR = r"D:\EMS_Data\CognitiveEMS_Datasets\North_Garden\DataForKeshara\23-10-2024"

# Define file pattern rules
FILE_PATTERNS = [
    # (include_ext, exclude_keyword)
    ("gsam2_deidentified.mp4", None),                  # GoPro
    ("synched_preview.mp4", None),                    # Smartphone
    ("_trimmed.mkv", "fps_converted"),                # Kinect
    ("sync_smartwatch.csv", None),                    # Smartwatch
    ("sync_depth_sensor.csv", None),                  # Depth sensor
]

# Ensure destination directory exists
os.makedirs(DEST_DIR, exist_ok=True)

# Traverse and apply copy rules
for root, _, files in os.walk(SRC_DIR):
    for file in files:
        for include_ext, exclude_key in FILE_PATTERNS:
            if file.endswith(include_ext) and (exclude_key is None or exclude_key not in file):
                src_path = Path(root) / file
                rel_path = src_path.relative_to(SRC_DIR)
                dest_path = Path(DEST_DIR) / rel_path

                # Create destination subdirectories
                os.makedirs(dest_path.parent, exist_ok=True)

                # Copy if destination doesn't exist or size differs
                if not dest_path.exists() or os.path.getsize(src_path) != os.path.getsize(dest_path):
                    print(f"Copying: {src_path} -> {dest_path}")
                    # shutil.copy2(src_path, dest_path)

print(f"[SUCCESS] All matching files copied to {DEST_DIR}, preserving structure.")
