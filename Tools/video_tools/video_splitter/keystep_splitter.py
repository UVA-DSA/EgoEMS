import os
import subprocess
import json
import argparse
import datetime

# Get command-line arguments
parser = argparse.ArgumentParser(description="Keystep video splitter script.")
parser.add_argument("--annotation_file_path", type=str, required=True, help="Path to the annotation file.")
parser.add_argument("--dataset_root", type=str, required=True, help="Root directory where the dataset will be saved.")
parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Train or Test split.")
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
if view == 'ego':   
    dataset_root = os.path.join(dataset_root, "ego")
elif view == 'exo':
    dataset_root = os.path.join(dataset_root, "exo")
elif view == 'ego+exo':
    dataset_root = os.path.join(dataset_root, "ego+exo")

# Check split and update the dataset root
print(f"[INFO] Checking split: {split}")
if split == "train":
    dataset_root = os.path.join(dataset_root, "train_root")
elif split == "test":
    dataset_root = os.path.join(dataset_root, "val_root")
print(f"{'*'*60}\n")

# Ensure the root directory exists
if not os.path.exists(dataset_root):
    print(f"[INFO] Creating root directory: {dataset_root}")
    os.makedirs(dataset_root)

# Function to create a unique ID for each clip
def create_clip_id(subject, scenario, trial, keystep, start_t, end_t, view):
    return f"{subject}_{scenario}_t{trial}_ks{keystep}_{start_t:.3f}_{end_t:.3f}_{view}"

# Iterate over each subject and trial
print(f"{'='*60}")
print("[INFO] Starting video processing for each subject and trial.")
print(f"{'='*60}\n")
for subject_data in data['subjects']:
    subject_id = subject_data['subject_id']
    print(f"{'*'*60}")
    print(f"[INFO] Processing subject: {subject_id}")
    
    for scenario in subject_data['scenarios']:
        scenario_id = scenario['scenario_id']
        print(f"  [INFO] Processing scenario: {scenario_id}")

        for trial_data in scenario['trials']:
            trial_id = trial_data['trial_id']
            print(f"  [INFO] Processing trial: {trial_id}")
            
            video_paths = []

            try:
                # Determine video paths based on view
                if view == 'ego':
                    video_path = trial_data['streams']['egocam_rgb_audio']['file_path']
                    video_paths.append((video_path, "ego"))
                elif view == 'exo':
                    video_path = trial_data['streams']['exocam_rgbd']['file_path']
                    video_paths.append((video_path, "exo"))
                elif view == 'ego+exo':  # Extract both ego and exo video clips
                    video_paths.append((trial_data['streams']['egocam_rgb_audio']['file_path'], "ego"))
                    video_paths.append((trial_data['streams']['exocam_rgbd']['file_path'], "exo"))

            except KeyError as e:
                print(f"[ERROR] Missing modality in trial data for subject {subject_id}, scenario {scenario_id}, trial {trial_id}: {e}")
                continue
            
            # Process each keystep
            for keystep in trial_data['keysteps']:
                label = keystep['label']
                start_t = keystep['start_t']
                end_t = keystep['end_t']
                keystep_id = keystep['class_id']

                for video_path, current_view in video_paths:
                    print(f"{'='*60}")

                    clip_id = create_clip_id(subject_id, scenario_id, trial_id, keystep_id, start_t, end_t, current_view)
                    
                    # Output directory based on label
                    output_dir = os.path.join(dataset_root, label)
                    if not os.path.exists(output_dir):
                        print(f"    [INFO] Creating directory for label '{label}': {output_dir}")
                        os.makedirs(output_dir)

                    # Output file path for the clip
                    output_clip = os.path.join(output_dir, f"{clip_id}.mp4")

                    # Check if the clip already exists
                    if os.path.exists(output_clip):
                        print(f"    [INFO] Clip already exists: {output_clip}")
                        continue

                    print(f"    [INFO] Generating clip for view: {current_view}, label: {label}, subject: {subject_id}, trial: {trial_id}, keystep: {keystep_id}")
                    
                    # FFmpeg command to extract the clip
                    ffmpeg_command = [
                        'ffmpeg',
                        '-ss', str(start_t),  # Start time
                        '-to', str(end_t),    # End time
                        '-i', video_path,     # Input video
                        '-c:v', 'libx264',    # Re-encode the video using H.264 to ensure proper extraction
                        '-c:a', 'aac',        # Re-encode audio to AAC
                        '-strict', 'experimental',  # Enable experimental features if needed
                        '-y',  # Overwrite output if exists
                        output_clip  # Output clip
                    ]
                    # Uncomment to execute FFmpeg command
                    subprocess.run(ffmpeg_command)
                    print(f"    [INFO] FFmpeg command: {' '.join(ffmpeg_command)}")
                    print(f"    [INFO] Generated clip: {output_clip}")
                    # break
                # break
            print(f"{'*'*60}\n")
            # break
        # break

print(f"{'='*60}")
print("[INFO] All clips have been generated.")
print(f"{'='*60}\n")
