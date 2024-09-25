import pandas as pd
import os
import sys

def recalculate_timestamps_cts(goPro_timestamp_file, offset_file):
    # Read the timestamp and offset files

    df = pd.read_csv(goPro_timestamp_file)
    offsets = pd.read_csv(offset_file)

    offset_value = offsets['offset'].iloc[0]

    # Extract GoPro file ID from the filename
    gopro_file_id = os.path.splitext(goPro_timestamp_file)[0].split("/")[-1]

    # Check if the GoPro file ID is in the offsets file
    if gopro_file_id not in offsets['file_id'].values:
        print(f"GoPro file id {gopro_file_id} is not in the offsets file")
        return

    # Get relevant offset values from the offset file
    offset_row = offsets[offsets['file_id'] == gopro_file_id].iloc[0]
    offset_value = offset_row['offset']

    print(f"Offset value {offset_value} for GoPro: {gopro_file_id}")
    # add the offset value to the first value of epoch column and save it as a new column
    df['recalculated_epoch'] = df['epoch'] - offset_value

    # make sure recalculated_epoch is in nanoseconds
    df['recalculated_epoch'] = df['recalculated_epoch'].astype('int64')

    # # Save the updated DataFrame back to a file (optional)
    df.to_csv(goPro_timestamp_file, index=False)
    # df.to_csv(goPro_timestamps, index=False)

if __name__ == "__main__":

    # read command line arguments
        # get cmd line arguments
    if len(sys.argv) < 2:
        exit("Usage: python kinect_trimmer.py <path_to_root_dir>")

    raw_data_path = sys.argv[1]
    goPro_timestamps = f"{raw_data_path}/goPro_timestamps/"

    #load offset file
    offset_file = f"{raw_data_path}/offsets.csv"
    
    # Iterate over all the GoPro timestamp files in the directory
    for file in os.listdir(goPro_timestamps):
        if file.endswith(".csv"):
            # Recalculate the timestamps for all GoPro files in the directory
            print("*"*50)
            print(f"Processing file: {file}")
            # if("GX010344" not in file):
            #     continue
            recalculate_timestamps_cts(os.path.join(goPro_timestamps, file), offset_file)
            print("*"*50)


