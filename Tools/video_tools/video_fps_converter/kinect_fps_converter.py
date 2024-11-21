import json
import os
import subprocess

import datetime
import sys



def convert_fps(filepath, target_fps):
    """Trims the video using ffmpeg based on start and end frames."""
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return
    # get the folder

    print("*"*50)

    output_folder = os.path.dirname(filepath)

    # extract the filename without the extension
    file_name = os.path.basename(filepath).split('.')[0]
    print(f"Converting FPS to {target_fps} for file {file_name}")
    
    output_file = os.path.join(output_folder, f"{file_name}_fps_converted.mkv")
    print(f"Output file: {output_file}")
 
    # If the file exists, rename it
    if os.path.exists(output_file):
        old_output_file = output_file.replace('.mkv', '_old.mkv')
        os.rename(output_file, old_output_file)
        print(f"Renamed existing file to: {old_output_file}")

    command = [
            'mkvmerge',
            '-o', output_file,  # Output file
            '--default-duration', f'0:{target_fps}fps',  # Split using timestamps
            filepath
        ]
    
    # # Run the ffmpeg command and let the output go to the terminal
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print(f"Trimming complete for {file_name}. Output saved to {output_file}.")
    else:
        print(f"Error trimming {file_name}. mkvmerge error:\n{result.stderr}")

    print("*"*50)


# Check if main
if __name__ == "__main__":
    
    # get cmd line arguments
    print(sys.argv)
    
    if len(sys.argv) < 2:
        exit("Usage: python kinect_fps_converter.py <path_to_root_dir>")

    raw_data_path = sys.argv[1]
    json_file = f"{raw_data_path}/depthCam_clip.json"

    # Load the JSON file
    with open(json_file, 'r') as f:
        video_data = json.load(f)

    # Process each entry in the JSON file
    for idx in video_data['filename'].keys():
        filename = video_data['filename'][idx]
        start_frame = video_data['start_frame'][idx]
        end_frame = video_data['end_frame'][idx]

        # if("anonymous/cardiac_arrest/2" not in filename):
        #     continue

        print(f"Processing file {filename}")
        
        try:
            # Trim the video using the given frames
            convert_fps(filename, target_fps=29.97)

        except Exception as e:
            print(f"Error processing video {filename}: {str(e)}")