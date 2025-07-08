import os
import csv
import cv2

# ============================
# Set these paths

video_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/"
train_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/audio_train_root"
val_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/audio_val_root"

# ============================

# Get all class names from both train and val
all_class_names = set()

for root in [train_root, val_root]:
    all_class_names.update([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

# Sort and create consistent mapping
class_names_sorted = sorted(all_class_names)
class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_names_sorted)}

print("Final class to label mapping:")
for k, v in class_to_label.items():
    print(f"{v}: {k}")

# Function to create annotation txt
def create_annotation_file(root_dir, output_file):
    lines = []

    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        label = class_to_label[cls_name]
        for fname in os.listdir(cls_path):
            if not fname.endswith(".npy"):
                continue

            # Build video file path
            video_fname = fname.replace(".npy", ".mp4")
            if root_dir == train_root:
                video_path = os.path.join(video_root, "train_root", cls_name, video_fname)
                relative_path = os.path.join("audio_train_root", cls_name, fname)
                
            else:
                video_path = os.path.join(video_root, "val_root", cls_name, video_fname)
                relative_path = os.path.join("audio_val_root", cls_name, fname)

            # Get number of frames
            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}, skipping...")
                continue
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            lines.append(f"{relative_path} {num_frames} {label}")
            print(f"Processed {relative_path} {num_frames} {label}")

    # Write txt file
    output_path = os.path.join(root_dir, output_file)
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Written {output_path}, {len(lines)} samples.")

# Create files for train and val
create_annotation_file(train_root, "train_audio.txt")
create_annotation_file(val_root, "val_audio.txt")
