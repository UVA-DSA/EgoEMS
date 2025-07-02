#!/bin/bash

# Variables
directory="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"  # Directory to search in
extension="mp4"  # File extension to search for
output_file="./cvpr_ego_file_paths.txt"   # Output file to store the paths

# Find files with the specified extension and write their full paths to the output file
find "$directory" -type f -name "*gsam2_deidentified.$extension" > "$output_file"

# Print a message when done
echo "File paths with extension .$extension written to $output_file"
