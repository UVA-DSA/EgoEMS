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
module load miniforge &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate egoexoems &&
module load ffmpeg &&

# Set root directories and synchronization offset path
root_dir="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Raw_Data/North_Garden/Sep_2024/Raw/"
repo_dir="/standard/UVA-DSA/Keshara/EgoExoEMS/Tools/"

# Set the day for which synchronization is being performed
day="20-09-2024"
# # # Step 1: Adjust GoPro timestamp offset
# echo "[INFO] Adjusting GoPro timestamp offset..."
# python "$repo_dir/synchronization/gopro_timestamp_adjuster.py" "$root_dir" "$day"
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Failed to adjust GoPro timestamp offset."
#     exit 1
# fi
# echo "[SUCCESS] GoPro timestamp adjustment completed."
# echo ""

# Ask the user to continue with the next steps
read -p "Press Enter to Generate Synchronization Metadata ..."

# Step 2: Generate synchronization metadata
echo "[INFO] Generating synchronization metadata..."
python "$repo_dir/synchronization/synchronization-v2.py" "$root_dir" "$day"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to generate synchronization metadata."
    exit 1
fi
echo "[SUCCESS] Synchronization metadata generated."
echo ""

# # # # Not needed anymore for this task.
# # # # Step 3: Convert Kinect frame rate to 29.97 FPS (if needed)
# # # echo "[INFO] Converting Kinect frame rate to 29.97 FPS..."
# # # python "$repo_dir/video_tools/video_fps_converter/kinect_fps_converter.py" "$root_dir"
# # # if [ $? -ne 0 ]; then
# # #     echo "[ERROR] Failed to convert Kinect frame rate."
# # #     exit 1
# # # fi
# # # echo "[SUCCESS] Kinect frame rate conversion completed."
# # # echo ""

read -p "Press Enter to Trim GoPro Recordings ..."

# # Step 4: Trim GoPro recordings
echo "[INFO] Trimming GoPro recordings..."
python -u "$repo_dir/video_tools/video_trimmer/gopro_trimmer.py" "$root_dir"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to trim GoPro recordings."
    exit 1
fi
echo "[SUCCESS] GoPro recordings trimmed."
echo ""

# # Ask the user to continue with the next steps
read -p "Press Enter to Trim Kinect Recordings ..."

# Step 5: Trim Kinect recordings
echo "[INFO] Trimming Kinect recordings..."
python "$repo_dir/video_tools/video_trimmer/kinect_trimmer.py" "$root_dir"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to trim Kinect recordings."
    exit 1
fi
echo "[SUCCESS] Kinect recordings trimmed."

# Ask the user to continue with the next steps
read -p "Press Enter to Create Side-by-Side Preview ..."

Step 6: Create a side-by-side preview of synchronized GoPro and Kinect videos
echo "[INFO] Creating side-by-side preview of synchronized videos..."
day="20-09-2024"
python "$repo_dir/synchronization/sync_clip_merger.py" "$root_dir" "$day"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create side-by-side preview."
    exit 1
fi
echo "[SUCCESS] Side-by-side preview created successfully."
echo ""

# # # Final message after successful completion of all tasks
echo "[INFO] All synchronization steps completed successfully."