import os
import subprocess
import json
import argparse
import datetime
import shutil

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

new_dataset_root = os.path.join(dataset_root, "final_exo_kinect_cpr_clips")
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

if not os.path.exists(new_dataset_root):
    print(f"[INFO] Creating new dataset root directory: {new_dataset_root}")
    os.makedirs(new_dataset_root)

# Function to create a unique ID for each clip
def create_clip_id(subject, trial, keystep, start_t, end_t, view):
    return f"{subject}_t{trial}_ks{keystep}_{start_t:.3f}_{end_t:.3f}_{view}"


generated_count = 0

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
        print(f"  [INFO] Scenario: {scenario_id}")
    
        for trial_data in scenario['trials']:
            trial_id = trial_data['trial_id']
            print(f"  [INFO] Processing trial: {trial_id}")
            
            video_paths = []

            # Determine video paths based on view
            # video_path = trial_data['streams']['exocam_rgbd']['file_path']

            original_video_path = f"/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/{subject_id}/{scenario_id}/{trial_id}/kinect/"
            # find the kinect video file in this directory
            if not os.path.exists(original_video_path):
                original_video_path = f"/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/{subject_id}/{scenario_id}/{trial_id}/Kinect/"

            for file in os.listdir(original_video_path):
                if file.endswith("trimmed.mkv") :
                    video_path = os.path.join(original_video_path, file)
                    print(f"    [INFO] Found video file: {video_path}")
                    break

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
                        # os.makedirs(output_dir)

                    # Output file path for the clip
                    output_clip = os.path.join(output_dir, f"{clip_id}.mp4")

                    print(f"    [INFO] Generating clip for view: {current_view}, label: {label}, subject: {subject_id}, trial: {trial_id}, keystep: {keystep_id}")
    

                    # Kinect clip 
                    output_clip = os.path.join(output_dir, f"{clip_id}.mkv")
                    # Format times into hh:mm:ss
                    start_time_formatted = str(datetime.timedelta(seconds=start_t))
                    end_time_formatted = str(datetime.timedelta(seconds=end_t))

                    # check if output_clip already exists and has size > 0
                    if os.path.exists(output_clip) and os.path.getsize(output_clip) > 0:
                        print(f"    [INFO] Clip already exists and is non-empty: {output_clip}")

                        # copy this file to the new dataset root
                        new_output_clip = os.path.join(new_dataset_root, f"{clip_id}.mkv")
                        print(f"    [INFO] Copying existing clip to new dataset root: {new_output_clip}")

                        # append the new_output_clip to a txt file
                        with open(os.path.join(new_dataset_root, "existing_clips.txt"), "a") as f:
                            f.write(f"{new_output_clip}\n")

                        shutil.copy2(output_clip, new_output_clip)
                        
                        generated_count += 1

                        continue

                    # MKVmerge command to extract the clip
                        # Execute the mkvmerge command
                    mkvmerge_command = [
                        'mkvmerge',
                        '-o', output_clip,
                        '--split', f'parts:{start_time_formatted}-{end_time_formatted}',
                        video_path
                    ]
                    print(f"    [CMD] {' '.join(mkvmerge_command)}")
                    # subprocess.run(mkvmerge_command)

                    print(f"    [INFO] Generated clip: {output_clip}")
                    # break
                # break
            print(f"{'*'*60}\n")
            # break
        # break

print(f"{'='*60}")
print(f"[INFO] {generated_count} clips have been generated.")
print(f"{'='*60}\n")
