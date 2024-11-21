#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Keystep video splitter  Converter"
#SBATCH --error="./logs/job-%j-keystep_video_splitter.err"
#SBATCH --output="./logs/job-%j-keystep_video_splitter.output"
#SBATCH --partition="standard"
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="anonymous"

module purge
source /home/anonymous/.bashrc
echo "[INFO] Running on node: $HOSTNAME"
module load ffmpeg
module load anaconda

# Set the folder where MKV files are located
annotation_file="/scratch/anonymous/compute/2024/repos/EgoExoEMS/Annotations/splits/trials/test_split_classification.json"
dataset_save_dir="/standard/storage/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format"
view="exo" # ego, exo, ego+exo
split="test" # train,  test

# Quote paths with spaces
python keystep_splitter.py --annotation_file_path "$annotation_file" --dataset_root "$dataset_save_dir" --split "$split" --view "$view" &&
echo done
