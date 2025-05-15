#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS New File Copy Task"
#SBATCH --error="./logs/job-%j-egoexoems_new_file_copy_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_new_file_copy_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge &&
source /home/cjh9fw/.bashrc &&
echo "[INFO] Running on node: $HOSTNAME" &&
conda activate egoexoems &&

python -u copy_new_files.py

echo "[SUCCESS] All files have been copied , preserving the directory structure"