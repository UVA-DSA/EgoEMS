#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS Video CPR Quality"
#SBATCH --error="./job-%j-cpr_video_script.err"
#SBATCH --output="./job-%j-cpr_video_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cqa3ym/.bashrc  &&
echo "$HOSTNAME" &&
conda activate lahiru-cpr-quality &&
cd /scratch/cqa3ym/repos/EgoExoEMS &&
python Benchmarks/CPR_quality/vision/estimate.py &&
echo "Done" &&
exit
