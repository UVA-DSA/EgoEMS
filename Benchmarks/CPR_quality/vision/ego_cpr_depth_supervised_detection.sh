#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Ego CPR Rate detection Task"
#SBATCH --error="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_supervised_depth_detection.err"
#SBATCH --output="/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/logs/job-%j-egoexoems_ego_supervised_depth_detection.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:1
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


# define variable for model_size
# model_type="DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type="DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type="MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)


# Define the root directory for the dataset and pass model_size variable
python -u Benchmarks/CPR_quality/vision/supervised_depth_estimate_window_ego.py  --model_type "$model_type"  --slurm_id "$SLURM_JOB_ID" &&

echo "[INFO] Ego supervised CPR depth detection task completed successfully." 
