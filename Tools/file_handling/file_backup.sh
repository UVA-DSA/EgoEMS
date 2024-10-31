#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS Backup"
#SBATCH --error="./logs/job-%j-backup_to_scratch.err"
#SBATCH --output="./logs/job-%j-backup_to_scratch.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

echo "$HOSTNAME" 
# Set the folder where MP4 files are located
input_folder="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025"
output_folder="/scratch/cjh9fw/Rivanna/2024/" # mew wars data 

cp -r "$input_folder" "$output_folder"

done
