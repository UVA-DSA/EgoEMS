#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS CPR Late Fusion Benchmark"
#SBATCH --error="./logs/job-%j-cpr_fusion_train_script.err"
#SBATCH --output="./logs/job-%j-cpr_fusion_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate wrist_keypoint &&

python -u fusion_train.py \
  --window_size 150 \
  --batch_size 1 \
  --job_id "$SLURM_JOB_ID"

echo "Done" &&
exit
