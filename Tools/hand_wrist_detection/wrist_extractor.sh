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
conda activate wrist_keypoint &&

# Define the root directory for the dataset
ROOT_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final" &&

python detect.py --root_dir "$ROOT_DIR" &&

echo "[INFO] Wrist keypoint detection task completed successfully." 
