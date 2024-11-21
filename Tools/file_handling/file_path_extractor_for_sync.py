import os
import csv

# Base path to traverse
base_path = "/standard/storage/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"

date_to_generate = ["20-09-2024"] 
# CSV file to write
output_csv = f"{base_path}/file_paths_for_sync.csv"

# Columns for the CSV file
columns = ['date', 'subject', 'procedure', 'trial', 'gopro_file_path', 'kinect_file_path']

# List to store rows of data
data = []

# Dictionary to store the GoPro and Kinect paths per trial
trial_dict = {}

# Traverse the directory structure
for root, dirs, files in os.walk(base_path):
    # Split the root path to extract date, subject, procedure, and trial information
    path_parts = root.split(os.sep)
    
    # Ensure the folder structure matches expectations
    if len(path_parts) >= 13:
        date = path_parts[8]
        subject = path_parts[9]
        procedure = path_parts[10]
        trial = path_parts[11]
        
        if(date not in date_to_generate):
            continue

        # Create a key for this particular trial
        trial_key = (date, subject, procedure, trial)
        print(f"Processing: {trial_key}")
        
        # Initialize trial entry in dictionary if not already present
        if trial_key not in trial_dict:
            trial_dict[trial_key] = {"GoPro": None, "Kinect": None}
        
        # Paths for GoPro and Kinect
        if 'GoPro' in root:
            gopro_file_path = root
            # Find the MP4 file
            for file in os.listdir(gopro_file_path):
                if file.endswith("_encoded.MP4"):
                    trial_dict[trial_key]['GoPro'] = os.path.join(gopro_file_path, file)
                    print(f"GoPro: {os.path.join(gopro_file_path, file)}")
                    gopro_file_id = file.split('.')[0]  # Extract file name without extension
                    # gopro_timestamp_path = f'{base_path}/goPro_timestamps/{gopro_file_id.split("_")[0]}.csv'
                    # copy the timestamp file to the same folder as the GoPro video
                    # print(f"\nCopying {gopro_timestamp_path} to {gopro_file_path}")
                    # # gopro paths has spaces in the string. fix it before cp command 
                    # os.system(f'cp "{gopro_timestamp_path}" "{gopro_file_path}"')
                    break

        elif 'Kinect' in root:
            kinect_file_path = root
            # Find the MKV file
            for file in os.listdir(kinect_file_path):
                if file.endswith(".mkv"):
                    trial_dict[trial_key]['Kinect'] = os.path.join(kinect_file_path, file)
                    break
        
# Now process the collected paths from trial_dict
for trial_key, paths in trial_dict.items():
    gopro_file_path = paths.get("GoPro")
    kinect_file_path = paths.get("Kinect")
    
    # Only add to data if both GoPro and Kinect paths exist
    if gopro_file_path and kinect_file_path:
        print("*"*50)
        date, subject, procedure, trial = trial_key
        data.append([date, subject, procedure, trial, gopro_file_path, kinect_file_path])
        print(f"GoPro: {gopro_file_path}")
        print(f"Kinect: {kinect_file_path}")

        print("*"*50)
# Write the data to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)  # Write header
    writer.writerows(data)  # Write data

print(f"CSV file generated: {output_csv}")