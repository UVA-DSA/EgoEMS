#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="Dataset Finalization Task"
#SBATCH --error="./logs/job-%j-egoexoems_finalize_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_finalize_task.output"
#SBATCH --partition="standard"
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
conda activate speech_deidentify &&
module load ffmpeg &&

# set root directory for the dataset
# root_dir="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized/"
# root_dir="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/"
root_dir="/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026/cars_1/chest_pain/0/"

echo "[INFO] PII deidentificatio for speech and transcript"
python -u speech_censor_twostage.py "$root_dir" 

echo "[INFO] PII deidentificatio for speech and transcript done"