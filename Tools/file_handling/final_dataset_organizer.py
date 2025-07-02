import os
import shutil
import csv

total_transfer_size = 0
total_transfer_count = 0

# Checkers for different folder types
def should_copy_audio(file_name):
    base, ext = os.path.splitext(file_name)
    return (ext == '.mp3' or ext == '.json') and base.endswith('deidentified')

def should_copy_gopro(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.mp4' and (base.endswith('gsam2_deidentified') or base.endswith('_720p_deidentified')):
        return True
    if ext == '.json':
        return True
    return False

def should_copy_kinect(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.mp4' and (base.endswith('kinect_rgb_stream_gsam2_deidentified') or base.endswith('_trimmed_rgb_stream_deidentified')):
        return True
    if ext == '.hdf5':
        return True
    return False

def should_copy_i3d(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.npy' and base.endswith('i3d'):
        return True
    return False

def should_copy_resnet(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.npy' and base.endswith('resnet'):
        return True
    return False

def should_copy_clip(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.npy' and base.endswith('clip'):
        return True
        
    return False

def should_copy_distance_sensor(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.csv' and base.endswith('sync_depth_sensor'):
        return True
    return False

def should_copy_smartwatch(file_name):
    base, ext = os.path.splitext(file_name)
    if ext == '.csv' and base.endswith('sync_smartwatch'):
        return True
    if ext == '.csv' and base.startswith('synchronized_smartwatch'):
        return True
    return False

def copy_structure(src_dir, dest_dir, csv_path):
    global total_transfer_size, total_transfer_count

    # Open CSV file to write logs
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['file_name', 'destination_path', 'file_size_bytes', 'file_size_mb', 'file_size_gb', 'file_type']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(src_dir):

            # print(f"Processing directory: {root}")
            rel_path = os.path.relpath(root, src_dir)
            dest_path = os.path.join(dest_dir, rel_path)

            folder_name = os.path.basename(root)
            if folder_name == "audio":
                filtered_files = [f for f in files if should_copy_audio(f)]
                file_type = "audio"

            elif folder_name == "GoPro" :
                filtered_files = [f for f in files if should_copy_gopro(f)]
                dest_path = dest_path.replace("GoPro", "ego")
                file_type = "ego"

            elif folder_name == "gopro":
                filtered_files = [f for f in files if should_copy_gopro(f)]
                dest_path = dest_path.replace("gopro", "ego")
                file_type = "ego"

            elif folder_name == "Kinect":
                filtered_files = [f for f in files if should_copy_kinect(f)]
                dest_path = dest_path.replace("Kinect", "exo")
                file_type = "exo"

            elif folder_name == "kinect":
                filtered_files = [f for f in files if should_copy_kinect(f)]
                dest_path = dest_path.replace("kinect", "exo")
                file_type = "exo"


            elif folder_name == "i3d_flow":
                filtered_files = [f for f in files if should_copy_i3d(f)]
                file_type = "i3d"

            elif folder_name == "i3d_rgb":
                filtered_files = [f for f in files if should_copy_i3d(f)]
                file_type = "i3d"

            elif folder_name == "resnet":
                filtered_files = [f for f in files if should_copy_resnet(f)]
                dest_path = dest_path.replace("resnet", "resnet_ego") #
                file_type = "resnet"

            elif folder_name == "resnet_exo":
                filtered_files = [f for f in files if should_copy_resnet(f)]
                file_type = "resnet"
            
            elif folder_name == "resnet_ego":
                filtered_files = [f for f in files if should_copy_resnet(f)]
                file_type = "resnet"

            elif folder_name == "resnet50":
                filtered_files = [f for f in files if should_copy_resnet(f)]
                dest_path = dest_path.replace("resnet50", "resnet_ego") # 
                file_type = "resnet"

            elif folder_name == "resnet50-exo":
                filtered_files = [f for f in files if should_copy_resnet(f)]
                dest_path = dest_path.replace("resnet50-exo", "resnet_exo") # 
                file_type = "resnet"
            

            elif folder_name == "clip":
                filtered_files = [f for f in files if should_copy_clip(f)]
                dest_path = dest_path.replace("clip", "clip_ego") #
                file_type = "clip"

            elif folder_name == "clip_ego":
                filtered_files = [f for f in files if should_copy_clip(f)]
                file_type = "clip"

            elif folder_name == "clip_exo":
                filtered_files = [f for f in files if should_copy_clip(f)]
                file_type = "clip"

            elif folder_name == "distance_sensor_data":
                filtered_files = [f for f in files if should_copy_distance_sensor(f)]
                file_type = "distance_sensor"

            elif folder_name == "smartwatch_data":
                filtered_files = [f for f in files if should_copy_smartwatch(f)]
                file_type = "smartwatch"

            elif folder_name == "BBOX_MASKS":
                # copy all files in BBOX_MASKS
                filtered_files = files
                file_type = "bbox_masks"
                
            else:
                continue

            os.makedirs(dest_path, exist_ok=True)

            for f in filtered_files:
                print("\n" + "-*" * 20)
                src_file = os.path.join(root, f)
                dest_file = os.path.join(dest_path, f)

                src_size = os.path.getsize(src_file)
                total_transfer_size += src_size
                total_transfer_count += 1

                print(f"Source file size: {src_size / (1024 * 1024):.2f} MB")
                # Uncomment to actually copy

                # check if the file already exists in the destination and match size
                if os.path.exists(dest_file):
                    dest_size = os.path.getsize(dest_file)
                    if dest_size == src_size:
                        print(f"File already exists and matches size: {dest_file}")
                        continue
                    else:
                        print(f"File exists but size differs: {dest_file} (src: {src_size}, dest: {dest_size})")

                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} \n→\nTo: {dest_file}")
                print("-*" * 20)

                # Write CSV row
                writer.writerow({
                    'file_name': f,
                    'file_type': file_type,
                    'destination_path': dest_file,
                    'file_size_bytes': src_size,
                    'file_size_mb': src_size / (1024 * 1024),
                    'file_size_gb': src_size / (1024 * 1024 * 1024),
                })

if __name__ == "__main__":
    src_directories = ["/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/organized","/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized", "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"]
    dest_directory = "/scratch/cjh9fw/aaai_2026_test/"
    csv_log_path = "transfer_log.csv"

    for src_directory in src_directories:
        print(f"Starting file structure copy from {src_directory} to {dest_directory}...")

        copy_structure(src_directory, dest_directory, csv_log_path)

        print(f"Total files copied: {total_transfer_count}")
        print(f"Total transfer size: {total_transfer_size / (1024 * 1024):.2f} MB : {total_transfer_size / (1024 * 1024 * 1024):.2f} GB")
        print(f"Log file created: {csv_log_path}")
        print("File structure copy completed.")
