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
CSV_PATH="/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Tools/synchronization/multi-responder-sync/opvrs_frame_positions_final.csv"

# ---------------------------
# Print job configuration for logging purposes
# ---------------------------
echo "Starting GoPro Video Sync"
echo "----------------------------------------"

# ---------------------------
# Run the synchronization script
# ---------------------------
# This command calls the Python script, which will call ffmpeg to produce
# new synced video clips named primary_synced.mp4, secondary_synced.mp4, and driver_synced.mp4.
python -u gopro-sync-clipper.py \
    "$CSV_PATH" 

# ---------------------------
# Final message
# ---------------------------
echo "✅ Sync complete"