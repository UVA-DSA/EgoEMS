#!/bin/bash

#SBATCH --job-name="EgoExoEMS Audio Classification"
#SBATCH --error="./logs/job-%j-test_classification.err"
#SBATCH --output="./logs/job-%j-test_classification.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge
module load anaconda
conda activate egoexoems
python -u keystep_classification.py --data_split test --train_annotation_path /scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_classification.json --task classification --observation_window None
