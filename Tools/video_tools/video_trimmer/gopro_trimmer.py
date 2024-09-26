import json
import os
import subprocess
import sys


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
    
    output_file = os.path.join(output_dir, f"{file_name}_trimmed.mp4")
    
    # Get the frame rate of the video
    fps = get_frame_rate(filepath)
    
    # Calculate start time and duration in seconds
    start_seconds = start_frame / fps
    duration_seconds = (end_frame - start_frame) / fps
    
    print(f"Trimming video: {filepath} from frame {start_frame} to {end_frame}")
    print(f"Start time: {start_seconds} seconds, Duration: {duration_seconds} seconds")
    
    # Execute ffmpeg command to trim the video
    # command = [
    #     'ffmpeg', '-i', filepath,
    #     '-ss', str(start_seconds),  # Start time in seconds
    #     '-t', str(duration_seconds), # Duration in seconds
    #     '-vcodec', 'libx264', '-acodec', 'aac',  # Re-encode video to H.264 and audio to AAC,
    #     '-c', 'copy',  # Copy video and audio codecs,
    #     '-map', '0',  # Map all streams from the input file,
    #     output_file
    # ]
    
    command = [
        'ffmpeg', '-i', filepath,
        '-ss', str(start_seconds),  # Start time in seconds
        '-t', str(duration_seconds), # Duration in seconds
        '-map_metadata', '0', '-map', '0:u',  # Re-encode video to H.264 and audio to AAC,
        '-c', 'copy',  # Copy video and audio codecs,
        '-y',  # Overwrite output file if it exists
        output_file
    ]

    # Run the ffmpeg command and let the output go to the terminal
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print(f"Trimming complete for {filepath}. Output saved to {output_file}.")

        # print number of frames in the output file
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1', output_file
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        num_frames = int(result.stdout.strip())
        print(f"Number of frames in the output file: {num_frames} \n\n")
    else:
        print(f"Error trimming {filepath}. ffmpeg error:\n{result.stderr}")


# Check if main
if __name__ == "__main__":
    
    # get cmd line arguments
    print(sys.argv)
    
    if len(sys.argv) < 2:
        exit("Usage: python gopro_trimmer.py <path_to_root_dir>")

    raw_data_path = sys.argv[1]
    json_file = f"{raw_data_path}/goPro_clip.json"

    # Load the JSON file
    with open(json_file, 'r') as f:
        video_data = json.load(f)

    # Process each entry in the JSON file
    for idx in video_data['filename'].keys():
        filename = video_data['filename'][idx]
        start_frame = video_data['start_frame'][idx]
        end_frame = video_data['end_frame'][idx]
        
        # if("debrah/cardiac_arrest/2" not in filename):
        #     continue
        try:
            # Trim the video using the given frames
            trim_video(filename, start_frame, end_frame)

        except Exception as e:
            print(f"Error processing video {filename}: {str(e)}")