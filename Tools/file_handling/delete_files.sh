#!/usr/bin/env bash

# Change this to your root directory
root_dir="/scratch/cjh9fw/aaai_2026_test/"

# Find all directories (including root)
find "$root_dir" -type d | while read -r dir; do
    # Check if both files exist in this directory
    file1=$(find "$dir" -maxdepth 1 -type f -name "*_synced_720p_deidentified.mp4" | head -n 1)
    file2=$(find "$dir" -maxdepth 1 -type f -name "*_synced_720p_gsam2_deidentified.mp4" | head -n 1)

    if [[ -n "$file1" && -n "$file2" ]]; then
        echo "Both files found in: $dir"
        echo "Deleting: $file1"
        rm "$file1"
    fi
done
