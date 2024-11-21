#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS Dataset Benchmark"
#SBATCH --error="./logs/job-%j-egoexoems_datasets.err"
#SBATCH --output="./logs/job-%j-egoexoems_datasets.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="anonymous"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
python -u sanity_check.py &&
echo "Done" &&
exit
