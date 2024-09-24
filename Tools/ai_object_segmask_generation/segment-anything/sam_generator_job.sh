#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS SAM Mask generator"
#SBATCH --error="./logs/job-%j-mask_generation_script.err"
#SBATCH --output="./logs/job-%j-mask_generation_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate sam &&
python -u sam_generator_script.py &&
echo "Done" &&
exit
