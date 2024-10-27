#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS Video-to-Numpy Extractor"
#SBATCH --error="./logs/job-%j-Video-to-Numpy-extractor.err"
#SBATCH --output="./logs/job-%j-Video-to-Numpy-extractor.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge 
source /home/cjh9fw/.bashrc 
echo "$HOSTNAME" 
module load ffmpeg 

# Set the folder where MP4 files are located
input_folder="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Lahiru/"

# Find all MP4 files in the folder and its subdirectories
find "$input_folder" -type f -name "*trimmed_deidentified.mp4" | while read input_video; do

    # Skip any file that contains "encoded" in its filename
    if [[ "$input_video" == *encoded* ]]; then
        echo "Skipping already encoded file: $input_video"
        continue
    fi
    # Extract filename without extension
    filename=$(basename -- "$input_video")
    filename="${filename%.*}"

    # Get the directory where the input video is located
    video_dir=$(dirname "$input_video")

    echo "****************************************************"
    echo "Processing GoPro file ($input_video)"

    # Set the output video path (same folder, same filename with -encoded.mp4 extension)
    output_frames_npy="$video_dir/${filename}_encoded.MP4"

    # Reencode the video using libx264
    echo "Extracting Video Frames from GoPro $input_video to npy format..."
    echo "Saving extracted frames to $output_frames_npy"
    

    # Capture exit code to check for errors
    if [ $? -eq 0 ]; then
        echo "Frame extraction complete: $output_frames_npy"

        # # Verify output file properties
        # output_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$output_frames_npy")
        # output_framerate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 "$output_frames_npy")
        
        # echo "Number of frames in output: $output_frames"
        # echo "Framerate of output: $output_framerate"
    else
        echo "Error during reencoding of $input_video."
    fi
    echo "****************************************************"

done
