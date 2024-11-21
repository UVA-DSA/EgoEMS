#!/bin/bash
module load ffmpeg

# Set the folder where AVI files are located
input_folder="/standard/storage/CognitiveEMS_Datasets/EMS_Interventions/videos/oxygen-bvm"

# Desired output settings
framerate=25  # Set the framerate explicitly

# Loop through all AVI files in the folder
for input_video in "$input_folder"/*.avi; do
    # Extract filename without extension
    filename=$(basename -- "$input_video")
    filename="${filename%.*}"
    
    # Set the output video path (same folder, same filename with .mp4 extension)
    output_video="$input_folder/$filename.mp4"

    # Get the number of frames in the input video
    num_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$input_video")
    echo "Number of frames in input ($input_video): $num_frames"

    # Check the codec of the input video
    # codec_info=$(ffmpeg -i "$input_video" 2>&1 | grep -i "mjpeg")
    codec_info=$(ffmpeg -i "$input_video" 2>&1 | grep -i "mpeg4")
    echo "Codec information for $input_video: $codec_info"
    
    # If the codec is Motion JPEG, reencode the video to MP4 format
    # if [[ $codec_info == *"mjpeg"* ]]; then
    if [[ $codec_info == *"mpeg4"* ]]; then
        echo "Reencoding $input_video from MJPEG to MP4 format..."
        
        # Reencode with the framerate, and preserve the number of frames
        ffmpeg -i "$input_video" -c:v libx264 -pix_fmt yuv420p -r $framerate -c:a aac -strict experimental "$output_video"
        
        # Verify output file properties
        output_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$output_video")
        output_framerate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 "$output_video")
        
        echo "Reencoding complete: $output_video"
        echo "Number of frames in output: $output_frames"
        echo "Framerate of output: $output_framerate"
    else
        echo "The input video ($input_video) is not in Motion JPEG format."
    fi
done
