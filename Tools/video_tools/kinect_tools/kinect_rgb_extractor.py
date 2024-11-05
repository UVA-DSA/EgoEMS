import os
import subprocess
import sys

# Function to find files with a specific extension in a given directory
def find_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

# Function to run ffmpeg command to combine videos
def run_ffmpeg(kinect_file, output_dir):
    """Run ffmpeg to extract Kinect RGB stream and combine it with GoPro video."""
    
    # Define paths
    kinect_file_name = os.path.basename(kinect_file).split('.')[0]
    kinect_rgb_path = os.path.join(output_dir, "Kinect", f"{kinect_file_name}_rgb_stream.mp4")

    # check if "kinect" or "Kinect" in the kinect_file path
    if kinect_file.find("kinect") != -1:
        kinect_rgb_path = os.path.join(output_dir, "kinect", f"{kinect_file_name}_rgb_stream.mp4")
    else:
        kinect_rgb_path = os.path.join(output_dir, "Kinect", f"{kinect_file_name}_rgb_stream.mp4")

    # if os.path.exists(output_video_path):
    #     print(f"[INFO] Synched Preview video already exists at {output_video_path}")
    #     return
    # Ensure output directories exist

    # Step 1: Extract RGB stream from Kinect MKV file
    extract_rgb_command = [
        "ffmpeg", "-i", kinect_file, "-map", "0:v:0", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-y", kinect_rgb_path
    ]
    print(extract_rgb_command)
    print(f"[INFO] Extracting RGB stream from Kinect file: {kinect_file}")
    subprocess.run(extract_rgb_command, check=True)


# Main script
if __name__ == "__main__":
    
    # Get command line arguments
    if len(sys.argv) < 1:
        exit("[ERROR] Usage: python sync_clip_merger.py <path_to_root_dir> ")

    raw_data_path = sys.argv[1]

    # Traverse the base directory for each participant's folder
    for root, dirs, files in os.walk(raw_data_path):
        # Split the path and check if it's at the level of 'subject/trial' (e.g., 'bryan/cardiac_arrest/0')
        path_parts = root.split(os.sep)
        # print(path_parts, len(path_parts))

        # We expect something like: [base, 'subject', 'cardiac_arrest', 'trial']
        if len(path_parts) == 10:  # Adjust the depth based on your folder structure
            kinect_path = os.path.join(root, "Kinect")
            
            # check if the Kinect folder exists
            if not os.path.exists(kinect_path):
                kinect_path = os.path.join(root, "kinect")


            # Check if GoPro and Kinect folders exist
            if not os.path.exists(kinect_path):
                print(f"[ERROR] Kinect folder not found in: {root}")
                continue

            print(f"[INFO] Processing trial: {root}")

            # Find GoPro and Kinect files
            kinect_files = find_files(kinect_path, "trimmed_final.mkv")

            if not kinect_files:
                kinect_files = find_files(kinect_path, "trimmed.mkv")

            if kinect_files:
                print("\n" + "*" * 50)
                print(f"[INFO] Processing folder: {root}")

                kinect_file_path = os.path.join(kinect_path, kinect_files[0])


                print(f"[INFO] Kinect file: {kinect_file_path}")

                print("[INFO] Running ffmpeg to merge videos...")
                # Run ffmpeg to combine videos
                run_ffmpeg( kinect_file_path, root)
                print("*" * 50)
            else:
                print(f"[ERROR] No Kinect files found in: {kinect_path}")
