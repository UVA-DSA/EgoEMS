#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="GoPro Synchronization Task"
#SBATCH --error="./logs/job-%j-egoexoems_sync_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_sync_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Directory with your GoPro videos
VIDEO_DIR="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-29/stroke/t2/GoPro/secondary/"
OUTPUT="GX_COMBINED.mp4"
TEMP_LIST="file_list.txt"

# Go to the video directory
cd "$VIDEO_DIR" || { echo "Directory not found: $VIDEO_DIR"; exit 1; }

# Create a list of files to combine (sorted for correct order)
ls -1v *.MP4 | while read -r filename; do
    echo "file '$filename'" >> "$TEMP_LIST"
done

# Combine videos using ffmpeg
ffmpeg -f concat -safe 0 -i "$TEMP_LIST" -c:v libx264 -preset slow -crf 23 -c:a aac -b:a 128k "$OUTPUT"

# Clean up
rm "$TEMP_LIST"

echo "✅ Combined video saved as ../$OUTPUT"
