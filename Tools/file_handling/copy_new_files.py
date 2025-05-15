import os
import shutil

# === CONFIGURATION ===
src_dir = "/scratch/cjh9fw/Rivanna/2025/egoexoems_data/EgoExoEMS_CVPR2025/Dataset/Final/"
dest_dir = "/scratch/cjh9fw/Rivanna/2025/egoexoems_data/EgoEMS_AAAI2025/Dataset/Final/"


suffixes = ["_encoded_trimmed.mp3", "bbox_annotations.json", ".jpg",".png","_clip.npy","sync_depth_sensor.csv","_720p.mp4","_flow.npy","_rgb.npy","rgb_stream_deidentified.mp4","depth_ir_data.hdf5","trimmed_resnet.npy","final_resnet.npy","sync_smartwatch.csv"] 

# =====================
src_dir = os.path.abspath(src_dir)
dest_dir = os.path.abspath(dest_dir)

for root, _, files in os.walk(src_dir):
    for file in files:
        if any(file.endswith(ext) for ext in suffixes):
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, rel_path)

            if "smartglass" in src_path:
                continue

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if not os.path.exists(dest_path) or os.path.getmtime(src_path) > os.path.getmtime(dest_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied/Updated: {src_path} → {dest_path}")
            # else:
            #     print(f"Skipped (up-to-date): {dest_path}")

print("✅ Done copying matching files.")
