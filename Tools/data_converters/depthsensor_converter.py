import csv
import os
import re



def load_timestamps(timestamps_file):
    with open(timestamps_file, 'r') as f:
        # Read all lines, stripping any extra whitespace
        return [line.strip() for line in f]

def convert_depthsensor_data(input_file,  output_file):
    # Load timestamps into a list
    
    with open(input_file, 'r') as file, open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Writing the header row
        writer.writerow([
            "Timestamp (ns)", "Range (mm)", "Sequence"
        ])
        
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for seq_num,line in enumerate(lines,start=1):
                parts = line.strip().split(" ")
                if len(parts) != 2:
                    print(f"Warning: Skipping line with unexpected format")
                    continue
                
                # Prepare a row with the matched `server_epoch_ms`
                row = [
                    parts[0], parts[1], seq_num
                ]
                
                # Write the row to the CSV file
                writer.writerow(row)
            
    print("[INFO] Conversion complete! Data saved to", output_file)



root_dir = "/standard/storage/EgoExoEMS_CVPR2025/Dataset/anonymous"

# iterate recursively through all files in the root directory
for root, dirs, files in os.walk(root_dir):
    if "arduino" in root:
        print("*"*10, "="*10, "*"*10)
        print(f"[INFO] Processing arduino recording: {root}")

        trial_path = "/".join(root.split("/")[:-1])
        print(f"[INFO] Trial path: {trial_path}")

        # new depthsensor folder
        new_depthsensor_folder = os.path.join(trial_path, "VL6180")
        if not os.path.exists(new_depthsensor_folder):
            os.makedirs(new_depthsensor_folder)


        original_depthsensor_timestamps_file = None
        for file in files:
            if file.endswith(".txt"):
                if("timestamps" in file):
                    original_depthsensor_timestamps_file = os.path.join(root, file)

        if(original_depthsensor_timestamps_file):
            print(original_depthsensor_timestamps_file)

            # new depthsensor file
            new_depthsensor_file = os.path.join(new_depthsensor_folder, "VL6180_readings.csv")
            
            print(new_depthsensor_file)

            convert_depthsensor_data(original_depthsensor_timestamps_file, new_depthsensor_file)

            # break

        print("*"*10, "="*10, "*"*10)

