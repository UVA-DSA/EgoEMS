#!/bin/bash

# --- this job will be run on any available node
# and will output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS File Copy Task"
#SBATCH --error="./logs/job-%j-egoexoems_file_copy_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_file_copy_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Define the source and destination directories
SRC_DIR="/standard/UVA-DSA/NIST EMS Project Data/"
DEST_DIR="/scratch/cjh9fw/Rivanna/2025/egoexoems_data/"

# Copy all files from source to destination
rsync -avh --progress --ignore-existing --size-only "$SRC_DIR" "$DEST_DIR"
echo "[SUCCESS] All files have been copied to $DEST_DIR, preserving the directory structure"