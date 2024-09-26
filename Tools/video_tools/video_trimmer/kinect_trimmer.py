import json
import os
import subprocess
import datetime
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
        print(f"[ERROR] File not found: {filepath}")
        return

    # Get the output folder and file name
    output_folder = os.path.dirname(filepath)
    file_name = os.path.basename(filepath).split('.')[0]
    output_file = os.path.join(output_folder, f"{file_name}_trimmed.mkv")

    # If the output file already exists, rename it
    if os.path.exists(output_file):
        old_output_file = output_file.replace('.mkv', '_old.mkv')
        os.rename(output_file, old_output_file)
        print(f"[INFO] Existing file renamed to: {old_output_file}")

    # Get the frame rate of the video
    fps = get_frame_rate(filepath)
    
    # Calculate start time and duration in seconds
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    
    # Format times into hh:mm:ss
    start_time_formatted = str(datetime.timedelta(seconds=start_seconds))
    end_time_formatted = str(datetime.timedelta(seconds=end_seconds))

    # Calculate the duration
    duration_frames = end_frame - start_frame
    duration_seconds = end_seconds - start_seconds

    # Print the details in a cleaner format
    print(f"\n[INFO] Trimming video: {filepath}")
    print(f"       Start: Frame {start_frame} ({start_time_formatted}, {start_seconds:.2f} seconds)")
    print(f"       End:   Frame {end_frame} ({end_time_formatted}, {end_seconds:.2f} seconds)")
    print(f"       Duration: {duration_frames} frames ({duration_seconds:.2f} seconds)\n")
    
    # Execute the mkvmerge command
    command = [
        'mkvmerge',
        '-o', output_file,
        '--split', f'parts:{start_time_formatted}-{end_time_formatted}',
        filepath
    ]
    
    print(f"[CMD] {' '.join(command)}")
    
    # Uncomment when ready to execute
    result = subprocess.run(command)
    if result.returncode == 0:
        print(f"[SUCCESS] Trimming completed for {file_name}. Output saved to {output_file}.")
    else:
        print(f"[ERROR] Trimming failed for {file_name}. Error:\n{result.stderr}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("[ERROR] Usage: python kinect_trimmer.py <path_to_root_dir>")

    raw_data_path = sys.argv[1]
    json_file = os.path.join(raw_data_path, "depthcam_clip.json")

    # Load the JSON file
    with open(json_file, 'r') as f:
        video_data = json.load(f)

    # Process each entry in the JSON file
    for idx in video_data['filename'].keys():
        filename = video_data['filename'][idx]
        start_frame = video_data['start_frame'][idx]
        end_frame = video_data['end_frame'][idx]

        print("\n" + "="*60)
        print(f"[INFO] Processing video file: {filename}")
        print("="*60)

        try:
            # Trim the video based on frames
            trim_video(filename, start_frame, end_frame)
        except Exception as e:
            print(f"[ERROR] An error occurred while processing {filename}: {str(e)}")

        print("="*60 + "\n")
