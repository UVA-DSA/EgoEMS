#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS CLIP Classification"
#SBATCH --error="./logs/job-%j-alldata_classification.err"
#SBATCH --output="./logs/job-%j-alldata_classification.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/zar8jw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
python -u clip_model_classification.py --data_split alldata --train_annotation_path /scratch/zar8jw/EgoExoEMS/Annotations/main_annotation.json --observation_window None&&
echo "Done" &&
exit
