#!/bin/bash

# Define the source and destination directories
SRC_DIR="/standard/storage/CognitiveEMS_Datasets/North_Garden/May_2024/DO_NOT_DELETE/smartglass_smartwatch"
# DEST_DIR="/standard/storage/CognitiveEMS_Datasets/North_Garden/May_2024/DO_NOT_DELETE/ego_camera/interventions"
DEST_DIR="/standard/storage/CognitiveEMS_Datasets/North_Garden/May_2024/DO_NOT_DELETE/smartwatch_imu"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"




# Find and copy all files  in the interventions directory to the destination directory
# Create the destination directory if it doesn't exist

# Find and copy all files ending with _clipped.mp4 inside the interventions folders, preserving the directory structure
# Find and copy all files ending with _clipped.mp4 inside the interventions folders, preserving the directory structure
find "$SRC_DIR" -type f -path "*/smartwatch_data/*" -name "*.csv" | while read -r file; do
  # Get the directory path of the file relative to the source directory
  rel_path="${file#$SRC_DIR/}"
  dest_dir="$DEST_DIR/$(dirname "$rel_path")"
  
  # Create the destination directory if it doesn't exist
  mkdir -p "$dest_dir"
  
  # Copy the file to the destination directory
  cp "$file" "$dest_dir"
done

echo "All smartwatch csv files in folders have been copied to $DEST_DIR, preserving the directory structure"
