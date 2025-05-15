#!/bin/bash

#SBATCH --job-name="FrameTS_Generator"
#SBATCH --error="./logs/job-%j-frame_ts_generator.err"
#SBATCH --output="./logs/job-%j-frame_ts_generator.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# ---------------------------
# Load necessary modules and activate the conda environment
# ---------------------------
module purge &&
module load miniforge &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate egoexoems &&
module load ffmpeg &&

# ---------------------------
# Specify path to the folder containing the synced video
# ---------------------------
VIDEO_FOLDER="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/03-28/stroke/t3/GoPro/secondary/"

# ---------------------------
# Set known reference frame and its timestamp (in microseconds)
# ---------------------------
# For example, if you know frame number 378 has a timestamp of 1,743,202,795,125 µs
REF_FRAME=378
REF_TIMESTAMP=1743202795125

# ---------------------------
# Print job configuration for logging purposes
# ---------------------------
echo "Starting Frame Timestamp Generation"
echo "Video Folder: ${VIDEO_FOLDER}"
echo "Reference Frame #: ${REF_FRAME}"
echo "Reference Timestamp (µs): ${REF_TIMESTAMP}"

# ---------------------------
# Find the synced video file in the specified folder.
# The search is non-recursive; if you need recursion, remove '-maxdepth 1'
# ---------------------------
VIDEO_FILE=$(find "${VIDEO_FOLDER}" -maxdepth 1 -type f -name "*synced.mp4")

if [ -z "$VIDEO_FILE" ]; then
    echo "[ERROR] No video file with suffix 'synced.mp4' found in ${VIDEO_FOLDER}"
    exit 1
fi

echo "[INFO] Processing video file: ${VIDEO_FILE}"

# ---------------------------
# Run the frame timestamp generation script.
# The Python script is expected to generate a CSV with each frame's number and its computed timestamp.
# ---------------------------
python -u gopro-timestamp-gen.py \
    "$VIDEO_FILE" \
    "$REF_FRAME" \
    "$REF_TIMESTAMP"

# ---------------------------
# Final message
# ---------------------------
echo "✅ Frame timestamp CSV generation complete"
