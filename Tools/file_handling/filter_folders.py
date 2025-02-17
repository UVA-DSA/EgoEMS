import os
import pandas as pd
import shutil

# Define the base directory and CSV file path
BASE_DIR = "/mnt/d/EgoExoEMS_CVPR2025/Verification/Timestamped_Audio_Transcripts/DataForKeshara/01-05-2025"
CSV_FILE = "./folders_to_be_filtered.csv"

# Load the CSV file
df = pd.read_csv(CSV_FILE)

# Create a set of valid paths from the CSV
valid_paths = {
    os.path.join(row["subject"], row["scenario"], str(row["trial"]))
    for _, row in df.iterrows()
}

print("[INFO] Valid paths:" , valid_paths)
# Walk through the base directory
for root, dirs, _ in os.walk(BASE_DIR, topdown=True):
    # Calculate the relative root path
    rel_root = os.path.relpath(root, BASE_DIR)


    if dirs:
        if(dirs[0] != "audio"):
            # print("[INFO] Skipping directory")
            # print("[INFO] Current root path:", rel_root)
            continue

    # Skip directories if the current root path is a valid path
    if rel_root in valid_paths or rel_root == ".":
        continue

    # Remove directories not part of the valid paths
    dirs_to_remove = [
        d for d in dirs
        if os.path.normpath(os.path.join(rel_root, d)) not in valid_paths
    ]

    # Remove the invalid directories
    for dir_to_remove in dirs_to_remove:
        print("[INFO] Removing directory: ", dir_to_remove, 'from Root:', root)
        print(f"Removing: {root}")
        shutil.rmtree(root)

print("[SUCCESS] Removed directories not listed in the CSV.")
