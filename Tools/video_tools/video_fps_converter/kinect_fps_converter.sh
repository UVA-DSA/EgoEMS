#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# job output
#SBATCH --job-name="EgoExoEMS Kinect FPS Converter"
#SBATCH --error="./logs/job-%j-kinect_fps_converter.err"
#SBATCH --output="./logs/job-%j-kinect_fps_converter.output"
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

# Set the folder where MKV files are located
input_folder="/standard/storage/CognitiveEMS_Datasets/anonymous/Sep_2024/Raw/19-09-2024/"
input_folder="/standard/storage/EgoExoEMS_CVPR2025/Dataset/anonymous/"

# Find all MKV files in the folder and its subdirectories
find "$input_folder" -type f -name "*.mkv" | while read input_video; do

    # Skip any file that contains "fps_converted" in its filename
    if [[ "$input_video" == *fps_converted* ]]; then
        echo "[INFO] Skipping already converted file: $input_video"
        continue
    fi

    # Extract filename without extension
    filename=$(basename -- "$input_video")
    filename="${filename%.*}"

    # Get the directory where the input video is located
    video_dir=$(dirname "$input_video")

    echo "****************************************************"
    echo "[INFO] Processing Kinect file: $input_video"

    # Set the output video path
    output_video="$video_dir/${filename}_fps_converted.mkv"

    # Reencode the video to 29.97 FPS using mkvmerge
    echo "[INFO] Converting frame rate of Kinect file to 29.97 FPS"
    echo "[INFO] Saving converted video to: $output_video"
    
    # mkvmerge -o "$output_video" --default-duration 0:29.97fps "$input_video"

    # Capture exit code to check for errors
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Conversion complete: $output_video"

    else
        echo "[ERROR] Failed to convert $input_video."
    fi
    echo "****************************************************"

done
