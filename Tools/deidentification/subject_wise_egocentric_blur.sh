#!/bin/bash

# --- This job will run on any available node and deidentify the GoPro dataset.
#SBATCH --job-name="Deidentify_EgoExoEMS_GoPro_dataset"
#SBATCH --error="logs/job-%j-tunnel.err"
#SBATCH --output="logs/job-%j-tunnel.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Define dataset directory and other paths
DATASET_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"
TARGET_SUBJECT="wa1"
TARGET_TRIAL="3"
script_path='./egoblur/EgoBlur/script/demo_ego_blur.py'
model_path='./egoblur/EgoBlur/weights/ego_blur_face.jit'

# Pretty print function for consistent, professional output
pretty_print() {
    printf "\n%-50s : %s\n" "$1" "$2"
}

# Load necessary modules and activate the environment
pretty_print "[$(date)] Status" "Loading environment modules and activating conda environment..."
module purge
module load anaconda

pretty_print "[$(date)] Status" "Creating conda environment from environment.yaml file..."
pretty_print "[$(date)] Status" "Activating ego_blur environment..."

conda activate ego_blur

pretty_print "[$(date)] Status" "Starting video deidentification process..."

# Loop through each GoPro folder in the dataset directory
for gopro_folder in "$DATASET_DIR"/*/*/*/GoPro/
do
  # Extract subject and trial from the folder path
  subject=$(basename "$(dirname "$(dirname "$(dirname "$gopro_folder")")")")
  trial=$(basename "$(dirname "$gopro_folder")")

  # Skip if the subject or trial doesn't match the target
  if [[ "$subject" != "$TARGET_SUBJECT" || "$trial" != "$TARGET_TRIAL" ]]; then
    pretty_print "[$(date)] Skipping Folder" "Subject: $subject, Trial: $trial (Not Target)"
    continue
  fi

  # Process only videos that match the pattern *_encoded_trimmed.mp4
  for file in "$gopro_folder"*_encoded_trimmed.mp4
  do
    if [ -f "$file" ]; then
      pretty_print "[$(date)] Processing Video" "$file"
      filename=$(basename "$file")
      new_filename="${filename%.mp4}_deidentified.mp4"
      output_path="$gopro_folder$new_filename"
      
      pretty_print "[$(date)] Output Path" "$output_path"
      
      # Run the Python deidentification script
      python $script_path --face_model_path $model_path --input_video_path "$file" --output_video_path "$output_path" --face_model_score_threshold 0.95
      
      if [ $? -eq 0 ]; then
        pretty_print "[$(date)] Success" "Successfully deidentified video: $file"
      else
        pretty_print "[$(date)] Error" "Error deidentifying video: $file. Check log for details." >&2
      fi
    else
      pretty_print "[$(date)] Warning" "No matching video file found at path: $file" >&2
    fi
  done
done

pretty_print "[$(date)] Status" "Deidentification process completed."
