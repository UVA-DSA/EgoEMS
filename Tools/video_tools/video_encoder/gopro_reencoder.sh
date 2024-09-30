#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS GoPro Reencoder"
#SBATCH --error="./logs/job-%j-gopro-reencoding.err"
#SBATCH --output="./logs/job-%j-gopro-reencoding.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge 
source /home/cjh9fw/.bashrc 
echo "$HOSTNAME" 
module load ffmpeg 

# Set the folder where MP4 files are located
input_folder="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/19-09-2024/"

# Find all MP4 files in the folder and its subdirectories
find "$input_folder" -type f -name "*.MP4" | while read input_video; do

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
    output_video="$video_dir/${filename}_encoded.MP4"

    # Reencode the video using libx264
    echo "Reencoding GoPro $input_video to libx264 format..."
    echo "Saving reencoded video to $output_video"
    
    # # Ensure there is a space after -i
    # ffmpeg -y -i "$input_video" -nostdin  -threads 16 -vcodec libx264 -acodec aac "$output_video"

    ffmpeg -y -i "$input_video" -nostdin  -map_metadata 0 -map 0:u -c copy "$output_video"
    # Capture exit code to check for errors
    if [ $? -eq 0 ]; then
        echo "Reencoding complete: $output_video"

        # # Verify output file properties
        # output_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$output_video")
        # output_framerate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 "$output_video")
        
        # echo "Number of frames in output: $output_frames"
        # echo "Framerate of output: $output_framerate"
    else
        echo "Error during reencoding of $input_video."
    fi
    echo "****************************************************"

done
