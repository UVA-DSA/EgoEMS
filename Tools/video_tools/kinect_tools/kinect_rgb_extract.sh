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
#SBATCH --account="anonymous"

# Load necessary modules and activate the conda environment
module purge &&
module load anaconda &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate cogems &&
module load ffmpeg &&

# Set root directories and synchronization offset path
root_dir="/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final"
repo_dir="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Tools"

echo "[INFO] Extracting Kinect RGB streams..."

python "$repo_dir/video_tools/kinect_tools/kinect_rgb_extractor.py" "$root_dir" 
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to extract kinect rgb stream."
    exit 1
fi
echo "[SUCCESS] Kinect RGB streams extracted."
echo "Done"
