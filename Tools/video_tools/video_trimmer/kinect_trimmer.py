import json
import os
import subprocess

import datetime

raw_data_path = "/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
json_file = f"{raw_data_path}/depthCam_clip.json"

# Load the JSON file
with open(json_file, 'r') as f:
    video_data = json.load(f)

def get_frame_rate(video_file):
    """Gets the frame rate of the video using ffprobe."""
    command = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=nokey=1:noprint_wrappers=1', video_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps_str = result.stdout.strip()

    # Convert frame rate string to numeric value (e.g. "30000/1001" -> 29.97002997)
    num, denom = map(int, fps_str.split('/'))
    return num / denom

def trim_video(filepath, start_frame, end_frame):
    """Trims the video using ffmpeg based on start and end frames."""

    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return
    # get the folder
    output_folder = os.path.dirname(filepath)

    # extract the filename without the extension
    file_name = os.path.basename(filepath).split('.')[0]
    
    output_dir = os.path.join("synchronized", output_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name}_trimmed.mkv")

    # Get the frame rate of the video
    fps = get_frame_rate(filepath)
    
    # Calculate start time and duration in seconds
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    
    # Convert seconds to hh:mm:ss format
    start_time_formatted = str(datetime.timedelta(seconds=start_seconds))
    end_time_formatted = str(datetime.timedelta(seconds=end_seconds))

    # Calculate the duration in frames and seconds
    duration_frames = end_frame - start_frame
    duration_seconds = end_seconds - start_seconds

    # Print the details
    print(f"Trimming video: {filepath} from frame {start_frame} to {end_frame}")
    print(f"Start time: {start_time_formatted} (seconds: {start_seconds}), End time: {end_time_formatted} (seconds: {end_seconds}), Duration: {duration_frames} frames ({duration_seconds} seconds)")
    # Execute mkvmerge command to trim the video
    command = [
            'mkvmerge',
            '-o', output_file,  # Output file
            '--split', f'parts:{start_time_formatted}-{end_time_formatted}',  # Split using timestamps
            filepath
        ]
    # Run the ffmpeg command and let the output go to the terminal
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print(f"Trimming complete for {file_name}. Output saved to {output_file}.")

        # Print number of frames in the output file
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1', output_file
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            num_frames = int(result.stdout.strip())
            print(f"Number of frames in the output file: {num_frames} \n\n")
        except:
            print(f"Error getting number of frames for {file_name}. ffmpeg error:\n{result.stderr}")
    else:
        print(f"Error trimming {file_name}. ffmpeg error:\n{result.stderr}")

# Process each entry in the JSON file
for idx in video_data['filename']:
    filename = video_data['filename'][idx]
    start_frame = video_data['start_frame'][idx]
    end_frame = video_data['end_frame'][idx]
    
    try:
        # Trim the video using the given frames
        trim_video(filename, start_frame, end_frame)

    except Exception as e:
        print(f"Error processing video {filename}: {str(e)}")
        # print stack trace
        import traceback
        traceback.print_exc()