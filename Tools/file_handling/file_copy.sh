#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS File Copy Task"
#SBATCH --error="./logs/job-%j-egoexoems_file_copy_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_file_copy_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Define the source and destination directories
SRC_DIR="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
DEST_DIR="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/SynchronizedTest"

# File extension to search for
GP_FILE_EXT="*encoded_trimmed.mp4" # GoPro
GP_AVOID_EXT="*NONE*" # Kinect recordings

SP_FILE_EXT="synched_preview.mp4" # SmartPhone

KW_FILE_EXT="*_trimmed.mkv" # Kinect recordings
KW_AVOID_EXT="*_fps_converted*" # Kinect recordings

SW_FILE_EXT="sync_smartwatch.csv" # SW recordings
DS_FILE_EXT="sync_depth_sensor.csv" # DS recordings


# Folder name pattern to search for (you can adjust this pattern)
FOLDER_PATTERN="*BBOX_MASKS*"

# Find and copy all files in the directories that match the folder name pattern
cd "$SRC_DIR" || exit 1  # Change to source directory or exit on failure
find . -type d -name "$FOLDER_PATTERN" -print0 | while IFS= read -r -d '' dir; do
  rsync -avh --progress --ignore-existing --size-only "$dir/" "$DEST_DIR/$dir"
done


# # copy synced preview files
# find . -type f -name "$SP_FILE_EXT" -print0 | rsync -avh --progress --ignore-existing --size-only --files-from=- --from0 ./ "$DEST_DIR/"

# # copy gopro files
# find . -type f -name "$GP_FILE_EXT" ! -name "$GP_AVOID_EXT" -print0 | rsync -avh --progress --ignore-existing --size-only --files-from=- --from0 ./ "$DEST_DIR/"

# copy kinect files
# find . -type f -name "$KW_FILE_EXT" ! -name "$KW_AVOID_EXT" -print0 | rsync -avh --progress --ignore-existing --size-only --files-from=- --from0 ./ "$DEST_DIR/"

# # copy smartwatch and depth sensor files
# find . -type f -name "$SW_FILE_EXT" ! -name "$AVOID_EXT" -print0  | rsync -avh --progress --files-from=- --from0 ./ "$DEST_DIR/"
# find . -type f -name "$DS_FILE_EXT" ! -name "$AVOID_EXT" -print0  | rsync -avh --progress --files-from=- --from0 ./ "$DEST_DIR/"

echo ""
echo "[SUCCESS] All sync files in folders have been copied to $DEST_DIR, preserving the directory structure"
