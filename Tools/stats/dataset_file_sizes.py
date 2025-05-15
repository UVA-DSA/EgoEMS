import os
import pandas as pd

# === CONFIGURATION ===
dest_dir = "/scratch/cjh9fw/Rivanna/2025/egoexoems_data/EgoEMS_AAAI2025/Dataset/Final/"
suffixes = [
    "_encoded_trimmed.mp3", "bbox_annotations.json", ".jpg", ".png", "_clip.npy",
    "sync_depth_sensor.csv", "_720p.mp4", "_flow.npy", "_rgb.npy",
    "rgb_stream_deidentified.mp4", "depth_ir_data.hdf5",
    "trimmed_resnet.npy", "final_resnet.npy", "sync_smartwatch.csv"
]


suffix_map = {s: (s) for s in suffixes}
size_by_type = {suffix: 0 for suffix in suffixes}

# Traverse and accumulate file sizes
for root, _, files in os.walk(dest_dir):
    for file in files:
        for suffix in suffixes:
            if file.endswith(suffix):
                path = os.path.join(root, file)
                size_by_type[suffix] += os.path.getsize(path)
                if(suffix == "_720p.mp4"):
                    print(f"File: {path}, Size: {os.path.getsize(path)} bytes")
                break  # Don't double count if multiple suffixes match

# Convert to GB and format
records = [
    {"File_type": suffix_map[s], "Total_size_GB": round(size / 1e9, 3)}
    for s, size in size_by_type.items() if size > 0
]

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("./stat_outputs/file_type_size_summary.csv", index=False)
print("✅ Size summary written to 'file_type_size_summary.csv'")
