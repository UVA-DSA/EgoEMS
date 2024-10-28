#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS SAM Mask generator"
#SBATCH --error="./logs/job-%j-mask_generation_script.err"
#SBATCH --output="./logs/job-%j-mask_generation_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate sam &&


# Set the folder where MP4 files are located
input_folder="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/19-09-2024/"
input_folder="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/"

# Find all MP4 files in the folder and its subdirectories
find "$input_folder" -type f -name "bbox_annotations.json" | while read bbox_annotation; do

    # Extract filename without extension
    filename=$(basename -- "$bbox_annotation")
    filename="${filename%.*}"

    # Get the directory where the input video is located
    trial_dir=$(dirname "$bbox_annotation")

    echo "****************************************************"
    echo "Processing Annotation file ($bbox_annotation)"


    # Reencode the video using libx264
    echo "Generating Segmentation Masks for $trial_dir"
    
    python -u sam_generator_script.py --bbox_json_path "$bbox_annotation" --bbox_dir "$trial_dir" &&

    # Capture exit code to check for errors
    if [ $? -eq 0 ]; then
        echo "Segmentation mask generation complete: $trial_dir"

    else
        echo "Error during mask generation of $trial_dir."
    fi
    echo "****************************************************"

echo "All mask generation tasks completed."
done
