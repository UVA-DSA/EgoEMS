#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="Wrist keypoint extractor Task"
#SBATCH --error="./logs/job-%j-egoexoems_wrist_keypoint_extraction.err"
#SBATCH --output="./logs/job-%j-egoexoems_wrist_keypoint_extraction.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Load necessary modules and activate the conda environment
module purge &&
module load miniforge &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate wrist_keypoint &&

# Define the root directory for the dataset
# ROOT_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Kinect_CPR_Clips/Final/exo_kinect_cpr_clips/test_root/chest_compressions" && # for exo kinect
ROOT_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/GoPro_CPR_Clips/ego_gopro_cpr_clips/train_root" && # for gopro ego
VIDEOS_LIST_TXT="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/GoPro_CPR_Clips/ego_gopro_cpr_clips/train_root/all_files.txt" &&

# python -u detect.py --root_dir "$ROOT_DIR" --view "ego"  && # for auto discovery of videos
python -u detect.py --root_dir "$ROOT_DIR" --view "ego" --video_list_txt "$VIDEOS_LIST_TXT"  &&

echo "[INFO] Wrist keypoint detection task completed successfully." 
