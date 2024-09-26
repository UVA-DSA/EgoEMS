import os
import subprocess
import sys

# Function to find files with a specific extension in a given directory
def find_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

# Function to run ffmpeg command to combine videos
def run_ffmpeg(gopro_file, kinect_file, output_dir):
    # Define paths
    gopro_path = gopro_file

    kinect_file_name = os.path.basename(kinect_file).split('.')[0]
    kinect_rgb_path = os.path.join(output_dir, "Kinect", f"{kinect_file_name}_rgb_stream.mp4")
    output_video_path = os.path.join(output_dir, "SynchedPreview", "synched_preview.mp4")


    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "Kinect"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "SynchedPreview"), exist_ok=True)

    # Step 1: Extract RGB stream from Kinect MKV file
    extract_rgb_command = [
        "ffmpeg", "-i", kinect_file, "-map", "0:v:0", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-y", kinect_rgb_path
    ]
    subprocess.run(extract_rgb_command, check=True)

    # Step 2: Combine GoPro and RGB stream into final output
    combine_streams_command = [
        "ffmpeg", "-i", gopro_path, "-i", kinect_rgb_path,
        "-filter_complex", "[0:v]scale=1920:1080,fps=29.97,setpts=PTS-STARTPTS[v0]; [1:v]scale=1920:1080,fps=29.97,setpts=PTS-STARTPTS[v1]; [v0][v1]hstack=inputs=2,format=yuv420p[v]",
        "-map", "[v]", "-map", "0:a?", "-shortest", "-y", output_video_path
    ]
    subprocess.run(combine_streams_command, check=True)


# Main script
# Check if main
if __name__ == "__main__":
    
    # get cmd line arguments

    if len(sys.argv) < 3:
        exit("Usage: python sync_clip_merger.py <path_to_root_dir> <date>")

    raw_data_path = sys.argv[1]
    day = sys.argv[2]

    raw_data_path = f"{raw_data_path}/{day}"

    # Traverse the base directory for each participant's folder
    for root, dirs, files in os.walk(raw_data_path):
        # Split the path and check if it's at the level of 'subject/trial' (e.g., 'bryan/cardiac_arrest/0')
        path_parts = root.split(os.sep)
        
        # We expect something like: [base, 'subject', 'cardiac_arrest', 'trial']
        if len(path_parts) == 12:  # Adjust the depth based on your folder structure
            
            gopro_path = os.path.join(root, "GoPro")
            kinect_path = os.path.join(root, "Kinect")

            # Check if GoPro and Kinect folders exist
            if not os.path.exists(gopro_path) or not os.path.exists(kinect_path):
                # print("GoPro or Kinect folder not found.")
                continue
            
            # Find GoPro and Kinect files
            gopro_files = find_files(gopro_path, "_trimmed.mp4")
            kinect_files = find_files(kinect_path , "_fps_converted_trimmed.mkv")
            

            if gopro_files and kinect_files:
                # if("debrah/cardiac_arrest/2" not in root):
                #     continue
                
                print("*" * 50)
                print(f"\nProcessing folder: {root}")

                gopro_file_path = os.path.join(gopro_path,gopro_files[0]) 
                kinect_file_path = os.path.join(kinect_path,kinect_files[0]) 

                print(f"GoPro file: {gopro_file_path}")
                print(f"Kinect file: {kinect_file_path}\n")

                print("Running ffmpeg to merge videos...")
                # # Run ffmpeg process
                run_ffmpeg(gopro_file_path, kinect_file_path, root)
                print("*" * 50)

