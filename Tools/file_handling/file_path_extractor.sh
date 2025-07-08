#!/bin/bash

# Directory to search in
dataset_root="/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026"

# Where to write the list of ego folders with both files
output_file="./partial_and_final_doubles.txt"

# clear out any old output
: > "$output_file"

# Find every 'ego' directory under the root
find "$dataset_root" -type d -name ego | while read -r ego_dir; do
  # Check for at least one *rgb_final.mp4 and one *rgb_partial.mp4 in that dir
  if ls "$ego_dir"/*_rgb_final.mp4 >/dev/null 2>&1 \
     && ls "$ego_dir"/*_rgb_partial.mp4 >/dev/null 2>&1; then
    echo "$ego_dir" >> "$output_file"
  fi
done

echo "Done: folders with both _rgb_final and _rgb_partial listed in $output_file"
