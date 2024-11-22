import os
import json
import paramiko
from dotenv import load_dotenv

# Function to generate transcript path. Change it for your own needs.
def generate_transcript_path(video_path):
    # Replace the last segment of the path
    segment = video_path.split('/')[-1]  # Get the last segment (video filename)
    filename_without_extension = segment.rsplit('.', 1)[0]  # Get filename without extension
    transcript_filename = f'gemini_{filename_without_extension}.json'  # Create transcript filename
    # Build the new path
    transcript_path = '/'.join(video_path.split('/')[:-2] + ['audio', transcript_filename])
    return transcript_path

# Function to download files from remote server
def download_files(video_paths, remote_host, username, password, local_dir):
    # Establish an SFTP connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_host, username=username, password=password)
    sftp = ssh.open_sftp()

    for video_path in video_paths:
        try:
            # Generate transcript path
            transcript_path = generate_transcript_path(video_path)

            # Define local paths for saving files
            local_video_path = os.path.join(local_dir, "VideoFiles", os.path.basename(video_path.strip()))
            local_transcript_path = os.path.join(local_dir, "AnnotationFiles", os.path.basename(transcript_path.strip()))

            # Download video file
            print(f"Downloading video: {video_path.strip()}\n")
            sftp.get(video_path.strip(), local_video_path)

            # Download transcript file
            print(f"Downloading transcript: {transcript_path.strip()}\n")
            sftp.get(transcript_path.strip(), local_transcript_path)
        except Exception as e:
            print(f"Could not load video path {video_path} with transcript path {transcript_path}")

    # Close SFTP connection
    sftp.close()
    ssh.close()

# Function to generate the required file format
def create_file_format(video_files, annotation_files):
    # Combine video and annotation paths into the required format
    return [{"video": video, "annotation": annotation} for video, annotation in zip(video_files, annotation_files)]

# Example function to list downloaded files
def list_downloaded_files(video_dir, annotation_dir):
    # List all video and annotation files in their respective directories
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    annotation_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith(".json")]
    
    # Sort files to ensure pairing consistency (assumes sorted filenames match)
    video_files.sort()
    annotation_files.sort()

    return video_files, annotation_files

# Main function to populate the JSON file
def populate_json_file(video_dir, annotation_dir, output_file):
    # Get lists of downloaded files
    video_files, annotation_files = list_downloaded_files(video_dir, annotation_dir)

    # Generate the structured format
    file_structure = create_file_format(video_files, annotation_files)

    # Write to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(file_structure, json_file, indent=4)

    print(f"JSON file created at: {output_file}")

if __name__ == "__main__":
    load_dotenv()
    remote_host = os.getenv("REMOTE_HOST")
    username = os.getenv("USERNAME") 
    password = os.getenv("PASSWORD")  
    target_dir = '.'

    with open("files.txt", "r") as f:
        video_paths = f.readlines()

    download_files(video_paths, remote_host, username, password, target_dir)

    video_directory = "VideoFiles"  # Directory where videos are stored
    annotation_directory = "AnnotationFiles"  # Directory where annotations are stored
    output_json_path = "tasks.json"  # Output JSON file path

    populate_json_file(video_directory, annotation_directory, output_json_path)
