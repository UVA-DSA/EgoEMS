import csv
import os
import re


# Set default values for wrist position and sensor type
wrist_position = "right"  # Assuming wrist position
sensor_type = "acc"       # Assuming accelerometer sensor


def load_timestamps(timestamps_file):
    with open(timestamps_file, 'r') as f:
        # Read all lines, stripping any extra whitespace
        return [line.strip() for line in f]

def convert_smartwatch_data(input_file, timestamps_file, output_file):
    # Load timestamps into a list
    timestamps = load_timestamps(timestamps_file)
    
    with open(input_file, 'r') as file, open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Writing the header row
        writer.writerow([
            "sw_epoch_ms", "wrist_position", "sensor_type",
            "value_X_Axis", "value_Y_Axis", "value_Z_Axis",
            "seq_num", "server_epoch_ms"
        ])
        
        # Regular expression to match the line pattern
        line_pattern = re.compile(r'^(\d+),(\d+),(\d+),(\[.*\]),(.+)$')
        
        # Reading and processing each line from the text file
        index = 0
        for seq_num, line in enumerate(file, start=1):
            # Skip empty lines
            parts_length = len(line.strip().split(","))
            if parts_length != 7:
                print(f"Warning: Skipping line with unexpected format at seq_num {seq_num}")
                continue
            if not line.strip():
                print(f"Warning: Skipping empty line at seq_num {seq_num}")
                continue
            
            # check if first part is the number related to accelerometer
            sensor = line.strip().split(",")[0]
            if sensor != "1":
                print(f"Warning: Skipping line with data not related accelerometer  at seq_num {seq_num}")
                continue


            # Match the line against the pattern
            match = line_pattern.match(line.strip())
            if not match:
                print(f"Warning: Skipping line with unexpected format at seq_num {seq_num}")
                continue
            
            print("[INFO] processing smartwatch data", sensor)

            # Extract matched groups
            sw_epoch_ms = match.group(3)
            values_str = match.group(4)
            
            try:
                values = eval(values_str)  # Convert the string list to an actual list
                if len(values) != 3:
                    raise ValueError("Expected three axis values (X, Y, Z).")
            except (SyntaxError, ValueError) as e:
                print(f"Error: Skipping line at seq_num {seq_num} due to parsing error: {e}")
                continue
            
            # Get server_epoch_ms from timestamps file using seq_num (1-based index)
            server_epoch_ms = timestamps[index - 1] if index - 1 < len(timestamps) else timestamps[-1]
            # Prepare a row with the matched `server_epoch_ms`
            row = [
                sw_epoch_ms, wrist_position, sensor_type,
                values[0], values[1], values[2],
                index, server_epoch_ms
            ]
            
            index += 1
            
            # Write the row to the CSV file
            writer.writerow(row)

    print("[INFO] Conversion complete! Data saved to", output_file)



root_dir = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Lahiru"

# iterate recursively through all files in the root directory
for root, dirs, files in os.walk(root_dir):
    if "smartwatch" in root:
        print("*"*10, "="*10, "*"*10)
        print(f"[INFO] Processing Smartwatch recording: {root}")

        trial_path = "/".join(root.split("/")[:-1])
        print(f"[INFO] Trial path: {trial_path}")

        # new smartwatch folder
        new_smartwatch_folder = os.path.join(trial_path, "smartwatch_data")
        if not os.path.exists(new_smartwatch_folder):
            os.makedirs(new_smartwatch_folder)


        original_smartwatch_file = None
        original_smartwatch_timestamps_file = None
        for file in files:
            if file.endswith(".txt"):

                if("timestamps" in file):
                    original_smartwatch_timestamps_file = os.path.join(root, file)
                else:
                    original_smartwatch_file = os.path.join(root, file)

        if(original_smartwatch_file and original_smartwatch_timestamps_file):
            print(original_smartwatch_timestamps_file,original_smartwatch_file )

            # new smartwatch file
            new_smartwatch_file = os.path.join(new_smartwatch_folder, "sw_data.csv")

            convert_smartwatch_data(original_smartwatch_file,original_smartwatch_timestamps_file, new_smartwatch_file)

            # break

        print("*"*10, "="*10, "*"*10)



# # Input file in text format and output CSV path
# input_file = 'path/to/old_text_file.txt'
# output_file = 'path/to/converted_sw_data.csv'
