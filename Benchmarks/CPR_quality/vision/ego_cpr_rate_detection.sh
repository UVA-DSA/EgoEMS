#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoCPR Rate detection Task"
#SBATCH --error="/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_cpr_rate.err"
#SBATCH --output="/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_cpr_rate.output"
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
cd /home/cjh9fw/Desktop/2024/repos/EgoExoEMS

# Define the root directory for the dataset
python -u Benchmarks/CPR_quality/vision/rate_estimate_window_ego.py  &&

echo "[INFO] Ego CPR rate detection task completed successfully." 
