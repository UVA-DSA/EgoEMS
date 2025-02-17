#!/bin/bash

#SBATCH --job-name="EgoExoEMS CLIP Classification"
#SBATCH --error="./logs/job-%j-test_classification.err"
#SBATCH --output="./logs/job-%j-test_classification.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge
module load anaconda
conda activate egoexoems
python clip_model_classification.py --data_split test --train_annotation_path /scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_classification.json --observation_window None
