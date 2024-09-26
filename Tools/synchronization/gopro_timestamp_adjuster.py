import pandas as pd
import os
import sys

def recalculate_timestamps_cts(gopro_timestamp_file,kinect_timestamp_file,sync_gopro_frame, sync_kinect_frame,sync_gopro_time, sync_kinect_time,sync_offsete):
    # Read the timestamp and offset files

    gopro_timestamps_df = pd.read_csv(gopro_timestamp_file)

    # Read the Kinect timestamps from txt file
    kinect_timestamps = pd.read_csv(kinect_timestamp_file, header=None, names=['timestamp'], dtype=str)

    # print("GoPro timestamps: ", gopro_timestamps_df.head())
    # print("Kinect timestamps: ", kinect_timestamps.head())


    # calculate the cts difference in gopro file and average
    cts_diff = gopro_timestamps_df['cts'].diff().mean()*1e6
    cts_diff = int(cts_diff)
    print("CTS difference in ns: ", cts_diff)

    # Assign sync_kinect_time to the recalculated_epoch column for the row corresponding to sync_gopro_frame
    gopro_timestamps_df.at[sync_gopro_frame, 'recalculated_epoch'] = sync_kinect_time
    
    # Create a range of indices relative to sync_gopro_frame
    backward_range = range(sync_gopro_frame - 1, -1, -1)
    forward_range = range(sync_gopro_frame + 1, len(gopro_timestamps_df))

    # Adjust backwards from sync_gopro_frame: keep subtracting cts_diff progressively
    for i in range(sync_gopro_frame - 1, -1, -1):
        gopro_timestamps_df.iloc[i, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] = (
            gopro_timestamps_df.iloc[i + 1, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] - cts_diff
        )

    # Adjust forwards from sync_gopro_frame: keep adding cts_diff progressively
    for i in range(sync_gopro_frame + 1, len(gopro_timestamps_df)):
        gopro_timestamps_df.iloc[i, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] = (
            gopro_timestamps_df.iloc[i - 1, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] + cts_diff
        )

    # Ensure recalculated_epoch is in nanoseconds and convert the column to int64
    gopro_timestamps_df['recalculated_epoch'] = gopro_timestamps_df['recalculated_epoch'].astype('int64')

    gopro_timestamps_df.to_csv(gopro_timestamp_file, index=False)

if __name__ == "__main__":

    # read command line arguments
        # get cmd line arguments
    if len(sys.argv) < 2:
        exit("Usage: python goPro_timestamp_adjuster.py <path_to_root_dir>")

    base_path = sys.argv[1]

    #load offset csv
    offset_file_path = f"{base_path}/sync_offset_data.csv"
    
    if not os.path.exists(offset_file_path):
        exit(f"Offset file not found at {offset_file_path}")

    offset_file = pd.read_csv(offset_file_path)

    # Iterate over all entries in the offset file
    for index, row in offset_file.iterrows():

        print("*"*50)

         # columns in offset file: sync_gopro_frame,sync_kinect_frame,sync_gopro_time,sync_kinect_time,sync_offset_gp-kinect,,gopro_file_path,gopro_timestamp_path,kinect_file_path,kinect_timestamp_path
        gopro_file_id = row['gopro_file_id']
        gopro_timestamp_file = row['gopro_timestamp_path']
        kinect_timestamp_file = row['kinect_timestamp_path']
        sync_offset = row['sync_offset_gp-kinect']
        sync_gopro_frame = row['sync_gopro_frame']
        sync_kinect_frame = row['sync_kinect_frame']
        sync_gopro_time = row['sync_gopro_time']
        sync_kinect_time = row['sync_kinect_time']

        # Check if the GoPro timestamp file exists
        if not os.path.exists(gopro_timestamp_file) or not os.path.exists(kinect_timestamp_file):
            exit(f"Timestamps folder not found at {gopro_timestamp_file} or {kinect_timestamp_file}")
        
        print(f"Processing GoPro recoding: {gopro_file_id}")
        print("GoPro sync frame: ", sync_gopro_frame)
        print("Kinect sync frame: ", sync_kinect_frame)
        print("GoPro sync time: ", sync_gopro_time)
        print("Kinect sync time: ", sync_kinect_time)
        print("Offset: ", sync_offset)

        recalculate_timestamps_cts(gopro_timestamp_file,kinect_timestamp_file,sync_gopro_frame, sync_kinect_frame,sync_gopro_time, sync_kinect_time,sync_offset)

        print("Timestamps recalculated for GoPro recording: ", gopro_file_id)
        print("*"*50)

        



