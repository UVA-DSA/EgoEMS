#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Wrist keypoint extractor Task"
#SBATCH --error="./logs/job-%j-egoexoems_wrist_keypoint_extraction.err"
#SBATCH --output="./logs/job-%j-egoexoems_wrist_keypoint_extraction.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="anonymous"

# Load necessary modules and activate the conda environment
module purge &&
module load anaconda &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate wrist_keypoint &&

# Define the root directory for the dataset
ROOT_DIR="/standard/storage/EgoExoEMS_CVPR2025/Dataset/Kinect_CPR_Clips/Final/exo_kinect_cpr_clips/test_root/chest_compressions" &&

python -u detect.py --root_dir "$ROOT_DIR" &&

echo "[INFO] Wrist keypoint detection task completed successfully." 
