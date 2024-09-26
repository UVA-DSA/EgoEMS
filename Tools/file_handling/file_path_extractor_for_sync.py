import os
import csv

# Base path to traverse
base_path = "/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"

# CSV file to write
output_csv = "output.csv"

# Columns for the CSV file
columns = ['date', 'subject', 'procedure', 'trial', 'gopro_file_path', 'kinect_file_path']

# List to store rows of data
data = []

# Traverse the directory structure
for root, dirs, files in os.walk(base_path):
    # Check if both GoPro and Kinect directories are present
    if 'GoPro' in root or 'Kinect' in root:
        # Split the root path to extract date, subject, procedure, and trial information
        path_parts = root.split(os.sep)
        
        # Ensure the folder structure matches expectations
        if len(path_parts) >= 9:
            date = path_parts[7]
            subject = path_parts[8]
            procedure = path_parts[9]
            trial = path_parts[10]
            
            # Paths for GoPro and Kinect
            if 'GoPro' in root:
                gopro_file_path = root
                kinect_file_path = os.path.join(os.path.dirname(root), 'Kinect')
            elif 'Kinect' in root:
                kinect_file_path = root
                gopro_file_path = os.path.join(os.path.dirname(root), 'GoPro')
            
            # Add the row of data if both paths exist
            if os.path.exists(gopro_file_path) and os.path.exists(kinect_file_path):
                data.append([date, subject, procedure, trial, gopro_file_path, kinect_file_path])

# Write the data to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)  # Write header
    writer.writerows(data)  # Write data

print(f"CSV file generated: {output_csv}")
