#!/bin/bash

#SBATCH --job-name="EgoExoEMS MTRSAP Benchmark"
#SBATCH --error="./logs/job-%j-mtrsap_train_script.err"
#SBATCH --output="./logs/job-%j-mtrsap_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&

# Use torchrun instead of torch.distributed.launch
torchrun --nproc_per_node=4 ddp_clip_retrieval.py --job_id "$SLURM_JOB_ID"

echo "Done" &&
exit
