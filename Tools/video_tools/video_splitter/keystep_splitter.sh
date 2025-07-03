#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# job output
#SBATCH --job-name="Keystep Video Splitter Converter"
#SBATCH --error="./logs/job-%j-keystep_video_splitter.err"
#SBATCH --output="./logs/job-%j-keystep_video_splitter.output"
#SBATCH --partition="standard"
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge
source /home/cjh9fw/.bashrc
echo "[INFO] Running on node: $HOSTNAME"
module load ffmpeg
module load anaconda

# Set the folder where MKV files are located
annotation_file="/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_train_split_classification.json"
dataset_save_dir="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format"
view="ego" # ego, exo, ego+exo
split="train" # train,  test

# Quote paths with spaces
python keystep_splitter.py --annotation_file_path "$annotation_file" --dataset_root "$dataset_save_dir" --split "$split" --view "$view" &&
echo done
