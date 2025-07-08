import os
import csv

modality = 'audio' # Change to 'video' or 'audio' as needed

# ============================
# Set these paths

if modality == 'video':
    train_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/train_root"
    val_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/val_root"
elif modality == 'audio':
    train_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/audio_train_root"
    val_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/audio_val_root"
# ============================

model = 'mmaction2'  # Change to 'mmaction2' or 'timesformer' as needed

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

# Function to create annotation CSV and text file
def create_annotation_file(root_dir, output_file_base):
    global model
    rows = []
    txt_lines = []
    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        label = class_to_label[cls_name]
        for fname in os.listdir(cls_path):
            if fname.endswith(".mp4") and modality == 'video':
                relative_path = os.path.join(cls_name, fname)
                if model == 'mmaction2':
                    relative_path_full = relative_path
                else:
                    # add train_root or val_root as prefix
                    if root_dir == train_root:
                        relative_path_full = os.path.join("train_root", relative_path)
                    else:
                        relative_path_full = os.path.join("val_root", relative_path)
                # Append to CSV rows
                rows.append([relative_path_full, label])
                # Append to txt lines
                txt_lines.append(f"{relative_path_full} {label}")
            elif fname.endswith(".wav") and modality == 'audio':
                relative_path = os.path.join(cls_name, fname)
                if model == 'mmaction2':
                    relative_path_full = relative_path
                else:
                    # add train_root or val_root as prefix
                    if root_dir == train_root:
                        relative_path_full = os.path.join("audio_train_root", relative_path)
                    else:
                        relative_path_full = os.path.join("val_root", relative_path)
                # Append to CSV rows
                rows.append([relative_path_full, label])
                # Append to txt lines
                txt_lines.append(f"{relative_path_full} {label}")

    # Save CSV
    csv_path = os.path.join(root_dir, output_file_base + ".csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    # Save TXT
    txt_path = os.path.join(root_dir, output_file_base + ".txt")
    with open(txt_path, "w") as f:
        for line in txt_lines:
            f.write(line + "\n")

    print(f"Written {csv_path} and {txt_path}, {len(rows)} samples.")

# Create files for train and val
if modality == 'video':
    create_annotation_file(train_root, "train")
    create_annotation_file(val_root, "val")
elif modality == 'audio':   
    create_annotation_file(train_root, "audio_train")
    create_annotation_file(val_root, "audio_val")

print(f"Total number of classes: {len(class_to_label)}")
