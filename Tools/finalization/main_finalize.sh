#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="Dataset Finalization Task"
#SBATCH --error="./logs/job-%j-egoexoems_finalize_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_finalize_task.output"
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

tools_dir="/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Tools"

# set root directory for the dataset
root_dir="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized/"

# step 1: compress ego videos to 720p and reduce size
echo "[INFO] Compressing ego videos to 720p and reducing size..."
bash "$tools_dir/video_tools/video_encoder/gopro_reencoder.sh" "$root_dir" 


# step 2: remove unnecessary files