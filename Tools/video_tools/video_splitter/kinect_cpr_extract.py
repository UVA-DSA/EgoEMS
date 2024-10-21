import os
import subprocess
import json
import argparse
import datetime

# Get command-line arguments
parser = argparse.ArgumentParser(description="Keystep video splitter script.")
parser.add_argument("--annotation_file_path", type=str, required=True, help="Path to the annotation file.")
parser.add_argument("--dataset_root", type=str, required=True, help="Root directory where the dataset will be saved.")
parser.add_argument("--split", type=str, required=True, choices=["train","val", "test"], help="Train, Val or Test split.")
parser.add_argument("--view", type=str, required=True, choices=["ego", "exo", "ego+exo"], help="View to extract clips from: ego, exo, or both.")

args = parser.parse_args()
annotation_file_path = args.annotation_file_path
dataset_root = args.dataset_root
view = args.view
split = args.split

print(f"\n{'='*60}")
print(f"[INFO] Starting the process with view: {view}, split: {split}")
print(f"[INFO] Loading annotation file from: {annotation_file_path}")
print(f"{'='*60}\n")

# Load the annotation file
with open(annotation_file_path, 'r') as file:
    data = json.load(file)

# Determine dataset root based on view
print(f"{'*'*60}")

print("[INFO] Setting up dataset root directory.")
if view != 'exo':
    raise ValueError(f"Invalid view: {view}")

dataset_root = os.path.join(dataset_root, "exo_kinect_cpr_clips")

# Check split and update the dataset root
print(f"[INFO] Checking split: {split}")
if split == "train":
    dataset_root = os.path.join(dataset_root, "train_root")
elif split == "val":
    dataset_root = os.path.join(dataset_root, "val_root")
elif split == "test":
    dataset_root = os.path.join(dataset_root, "test_root")
else:
    raise ValueError(f"Invalid split: {split}")
print(f"{'*'*60}\n")

# Ensure the root directory exists
if not os.path.exists(dataset_root):
    print(f"[INFO] Creating root directory: {dataset_root}")
    os.makedirs(dataset_root)

# Function to create a unique ID for each clip
def create_clip_id(subject, trial, keystep, start_t, end_t, view):
    return f"{subject}_t{trial}_ks{keystep}_{start_t:.3f}_{end_t:.3f}_{view}"

# Iterate over each subject and trial
print(f"{'='*60}")
print("[INFO] Starting video processing for each subject and trial.")
print(f"{'='*60}\n")
for subject_data in data['subjects']:
    subject_id = subject_data['subject_id']
    print(f"{'*'*60}")
    print(f"[INFO] Processing subject: {subject_id}")
    
    for trial_data in subject_data['trials']:
        trial_id = trial_data['trial_id']
        print(f"  [INFO] Processing trial: {trial_id}")
        
        video_paths = []

        # Determine video paths based on view
        video_path = trial_data['streams']['exocam_rgbd']['file_path']
        video_paths.append((video_path, "exo"))

        # Process each keystep
        for keystep in trial_data['keysteps']:
            label = keystep['label']
            start_t = keystep['start_t']
            end_t = keystep['end_t']
            keystep_id = keystep['class_id']

            if(label != "chest_compressions"):
                continue

            for video_path, current_view in video_paths:
                print(f"{'='*60}")

                clip_id = create_clip_id(subject_id, trial_id, keystep_id, start_t, end_t, current_view)
                
                # Output directory based on label
                output_dir = os.path.join(dataset_root, label)
                if not os.path.exists(output_dir):
                    print(f"    [INFO] Creating directory for label '{label}': {output_dir}")
                    os.makedirs(output_dir)

                # Output file path for the clip
                output_clip = os.path.join(output_dir, f"{clip_id}.mp4")

                print(f"    [INFO] Generating clip for view: {current_view}, label: {label}, subject: {subject_id}, trial: {trial_id}, keystep: {keystep_id}")
  

                # Kinect clip 
                output_clip = os.path.join(output_dir, f"{clip_id}.mkv")
                # Format times into hh:mm:ss
                start_time_formatted = str(datetime.timedelta(seconds=start_t))
                end_time_formatted = str(datetime.timedelta(seconds=end_t))

                # MKVmerge command to extract the clip
                    # Execute the mkvmerge command
                mkvmerge_command = [
                    'mkvmerge',
                    '-o', output_clip,
                    '--split', f'parts:{start_time_formatted}-{end_time_formatted}',
                    video_path
                ]
                print(f"    [CMD] {' '.join(mkvmerge_command)}")
                subprocess.run(mkvmerge_command)

                print(f"    [INFO] Generated clip: {output_clip}")
                # break
            # break
        print(f"{'*'*60}\n")
        # break
    # break

print(f"{'='*60}")
print("[INFO] All clips have been generated.")
print(f"{'='*60}\n")
