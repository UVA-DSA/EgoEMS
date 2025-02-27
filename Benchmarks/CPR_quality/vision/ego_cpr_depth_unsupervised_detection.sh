#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Ego CPR Rate detection Task"
#SBATCH --error="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_unsupervised_depth_detection.err"
#SBATCH --output="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_unsupervised_depth_detection.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Load necessary modules and activate the conda environment
module purge &&
module load miniforge &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate wrist_keypoint &&

# change directory to the root of the repository
cd /scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS

# Define the root directory for the dataset
python -u Benchmarks/CPR_quality/vision/unsupervised_depth_estimate_window_ego.py  &&

echo "[INFO] Ego unsupervised CPR depth detection task completed successfully." 
