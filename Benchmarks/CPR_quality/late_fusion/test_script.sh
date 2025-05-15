#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS CPR Late Fusion Benchmark"
#SBATCH --error="./logs/job-%j-cpr_fusion_test_script.err"
#SBATCH --output="./logs/job-%j-cpr_fusion_test_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate wrist_keypoint &&

python -u fusion_test.py \
  --window_size 150 \
  --batch_size 1 \
  --weights_file ./weights/3817147_fusion_weights.txt

echo "Done" &&
exit
