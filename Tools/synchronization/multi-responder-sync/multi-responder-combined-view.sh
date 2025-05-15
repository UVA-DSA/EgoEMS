#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="GoPro Video Sync"
#SBATCH --error="./logs/job-%j-egoexoems_sync_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_sync_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Load necessary modules and activate the conda environment
module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate egoexoems &&
module load ffmpeg &&

# ---------------------------
# Specify paths to the video files to be synced
# ---------------------------
# IMPORTANT: Replace these with the full paths including filenames (e.g. primary.mp4)
PRIMARY_VIDEO_PATH="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-29/stroke/t2/GoPro/primary/GX010020_synced.mp4"
SECONDARY_VIDEO_PATH="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-29/stroke/t2/GoPro/secondary/GX_COMBINED_synced.mp4"
DRIVER_VIDEO_PATH="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-29/stroke/t2/GoPro/driver/GX_COMBINED_synced.mp4"

OUTPUT_VIDEO_PATH="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-29/stroke/t2/GoPro/combined_view.mp4" 

# ---------------------------
# Print job configuration for logging purposes
# ---------------------------
echo "Starting GoPro Video Combined View"
echo "Primary Video: ${PRIMARY_VIDEO_PATH}"
echo "Secondary Video: ${SECONDARY_VIDEO_PATH}"
echo "Driver Video: ${DRIVER_VIDEO_PATH}"



ffmpeg -y \
  -i "$PRIMARY_VIDEO_PATH" \
  -i "$SECONDARY_VIDEO_PATH" \
  -i "$DRIVER_VIDEO_PATH" \
  -filter_complex "\
    [0:v]scale=-2:1080:force_original_aspect_ratio=decrease,pad=w=iw:h=1080:x=(ow-iw)/2:y=(oh-ih)/2[v0]; \
    [1:v]scale=-2:1080:force_original_aspect_ratio=decrease,pad=w=iw:h=1080:x=(ow-iw)/2:y=(oh-ih)/2[v1]; \
    [2:v]scale=-2:1080:force_original_aspect_ratio=decrease,pad=w=iw:h=1080:x=(ow-iw)/2:y=(oh-ih)/2[v2]; \
    [v0][v1][v2]hstack=inputs=3[v]" \
  -map "[v]" -map 0:a? \
  -c:v libx264 -crf 23 -preset slow \
  -c:a aac -b:a 128k \
  "$OUTPUT_VIDEO_PATH"

  # ---------------------------
# Final message
# ---------------------------
echo "✅ COMBINATION complete"
