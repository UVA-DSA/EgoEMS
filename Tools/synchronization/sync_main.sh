#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Synchronization Task"
#SBATCH --error="./logs/job-%j-egoexoems_sync_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_sync_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Load necessary modules and activate the conda environment
module purge &&
module load anaconda &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate cogems &&
module load ffmpeg &&

# Set root directories and synchronization offset path
root_dir="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
sync_offset_dir="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/sync_offsets/2024-09-26_18-03-36"
repo_dir="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Tools"

# # Step 1: Adjust GoPro timestamp offset
# echo "[INFO] Adjusting GoPro timestamp offset..."
# python "$repo_dir/synchronization/gopro_timestamp_adjuster.py" "$sync_offset_dir"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to adjust GoPro timestamp offset."
#     exit 1
# fi
# echo "[SUCCESS] GoPro timestamp adjustment completed."
# echo ""

# # Step 2: Generate synchronization metadata
# echo "[INFO] Generating synchronization metadata..."
# python "$repo_dir/synchronization/synchronization-v2.py" "$root_dir" "$sync_offset_dir"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to generate synchronization metadata."
#     exit 1
# fi
# echo "[SUCCESS] Synchronization metadata generated."
# echo ""

# # Step 3: Convert Kinect frame rate to 29.97 FPS (if needed)
# echo "[INFO] Converting Kinect frame rate to 29.97 FPS..."
# python "$repo_dir/video_tools/video_fps_converter/kinect_fps_converter.py" "$root_dir"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to convert Kinect frame rate."
#     exit 1
# fi
# echo "[SUCCESS] Kinect frame rate conversion completed."
# echo ""

# # Step 4: Trim GoPro recordings
# echo "[INFO] Trimming GoPro recordings..."
# python "$repo_dir/video_tools/video_trimmer/gopro_trimmer.py" "$root_dir"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to trim GoPro recordings."
#     exit 1
# fi
# echo "[SUCCESS] GoPro recordings trimmed."
# echo ""

# # Step 5: Trim Kinect recordings
# echo "[INFO] Trimming Kinect recordings..."
# python "$repo_dir/video_tools/video_trimmer/kinect_trimmer.py" "$root_dir"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to trim Kinect recordings."
#     exit 1
# fi
# echo "[SUCCESS] Kinect recordings trimmed."
# echo ""

# Step 6: Create a side-by-side preview of synchronized GoPro and Kinect videos
echo "[INFO] Creating side-by-side preview of synchronized videos..."
python "$repo_dir/synchronization/sync_clip_merger.py" "$root_dir" "$day"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create side-by-side preview."
    exit 1
fi
echo "[SUCCESS] Side-by-side preview created successfully."
echo ""

# Final message after successful completion of all tasks
echo "[INFO] All synchronization steps completed successfully."
