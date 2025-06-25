#!/bin/bash

# --- This job will run on any available node and deidentify the GoPro dataset.
#SBATCH --job-name="Deidentify_EgoExoEMS_GoPro_dataset"
#SBATCH --error="logs/egoblur_task_%j.err"
#SBATCH --output="logs/egoblur_task_%j.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G  
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

# Define dataset directory and other paths
DATASET_DIR="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"
DATASET_DIR="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized/cars_1/chest_pain/"
# DATASET_DIR="/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized"
script_path='./egoblur/EgoBlur/script/efficient_ego_blur.py'
model_path='./egoblur/EgoBlur/weights/ego_blur_face.jit'

# Pretty print function for consistent, professional output
pretty_print() {
    printf "\n%-50s : %s" "$1" "$2"
}

# Load necessary modules and activate the environment
pretty_print "[$(date)] Status" "Loading environment modules and activating conda environment..."
module purge
module load miniforge

pretty_print "[$(date)] Status" "Creating conda environment from environment.yaml file..."
pretty_print "[$(date)] Status" "Activating ego_blur environment..."

conda activate ego_blur

pretty_print "[$(date)] Status" "Starting video deidentification process..."

# Loop through each GoPro folder in the dataset directory
# for gopro_folder in "$DATASET_DIR"/*/*/*/GoPro/
# for gopro_folder in "$DATASET_DIR"/*/*/*/gopro/ # adjusted to match lahirus data
for gopro_folder in "$DATASET_DIR"/*/GoPro/
do
  # # Check if any deidentified video already exists in the GoPro folder
  # if ls "$gopro_folder"/*_deidentified.mp4 1> /dev/null 2>&1; then
  #   pretty_print "[$(date)] Skipping Folder" "Deidentified video already exists in folder: $gopro_folder"
  #   continue
  # fi

  # Process each video in the GoPro folder if no deidentified video is found
  # for file in "$gopro_folder"*encoded_trimmed.mp4
  for file in "$gopro_folder"*synced_720p.mp4
  # for file in "$gopro_folder"*trimmed_720p.mp4
  do
    if [ -f "$file" ]; then
      pretty_print "----------------------" 
      pretty_print "[$(date)] Processing Video" "$file"
      filename=$(basename "$file")
      new_filename="${filename%.mp4}_deidentified.mp4"
      output_path="$gopro_folder$new_filename"
      
      pretty_print "[$(date)] Output Path" "$output_path"

      # Check if the output file already exists
      # if [ -f "$output_path" ]; then
      #   pretty_print "[$(date)] Skipping" "Deidentified video already exists: $output_path"
      #   continue
      # fi
      
      # # Run the Python deidentification script
      python -u $script_path --face_model_path $model_path --input_video_path "$file" --output_video_path "$output_path" --face_model_score_threshold 0.8
      
      if [ $? -eq 0 ]; then
        pretty_print "[$(date)] Success" "Successfully deidentified video: $file"
      else
        pretty_print "[$(date)] Error" "Error deidentifying video: $file. Check log for details." >&2
      fi


    else
      pretty_print "[$(date)] Warning" "No video file found at path: $file" >&2
    fi

    pretty_print "----------------------" 


  done


done

pretty_print "[$(date)] Status" "Deidentification process completed."
