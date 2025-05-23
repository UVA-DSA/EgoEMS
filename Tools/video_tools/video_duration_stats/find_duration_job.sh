#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Video Duration Stats"
#SBATCH --error="./logs/job-%j-vid-duration.err"
#SBATCH --output="./logs/job-%j-vid-duration.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&

# Set the folder where MP4 files are located
input_folder="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/organized" # mew wars data 

python -u find_duration.py --input_folder "$input_folder" --output_csv "./OPVRS_video_duration_stats.csv"

echo "Video duration stats completed."