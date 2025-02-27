#!/bin/bash

# Base directory where files are located
base_directory="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"

# Find incorrectly renamed files
find "$base_directory" -type f -name "*_trimmed_final_rgb_stream.mp4" | while read -r file; do
    # Extract the directory path of the file
    file_dir=$(dirname "$file")

    # Extract subject and trial from the correct directory levels
    path_parts=(${file//\// })  # Split path into an array by "/"
    subject="${path_parts[7]}"   # Subject (e.g., P16, ng2, ms1)
    trial="${path_parts[9]}"     # Trial (e.g., s5, t1)

    # Construct the correct filename
    correct_filename="${subject}_${trial}_trimmed_final_rgb_stream.mp4"
    correct_filepath="$file_dir/$correct_filename"

    # Rename file carefully
    # mv "$file" "$correct_filepath"
    echo "Fixed: $file -> $correct_filepath"
done

echo "✅ Renaming correction completed successfully."
