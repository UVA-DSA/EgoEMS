import pandas as pd
import os
import sys

def recalculate_timestamps_cts(gopro_timestamp_file, kinect_timestamp_file, sync_gopro_frame, sync_kinect_frame, sync_gopro_time, sync_kinect_time, sync_offset):
    """Recalculate timestamps for GoPro based on synchronization with Kinect."""
    
    # Read GoPro timestamps
    gopro_timestamps_df = pd.read_csv(gopro_timestamp_file)

    # Read Kinect timestamps
    kinect_timestamps = pd.read_csv(kinect_timestamp_file, header=None, names=['timestamp'], dtype=str)

    # Calculate the average CTS difference in GoPro file
    cts_diff = gopro_timestamps_df['cts'].diff().mean() * 1e6  # Convert to nanoseconds
    cts_diff = int(cts_diff)
    print(f"[INFO] CTS difference in ns: {cts_diff}")

    # Assign Kinect sync time to GoPro recalculated_epoch for the sync frame
    gopro_timestamps_df.at[sync_gopro_frame, 'recalculated_epoch'] = sync_kinect_time

    # Adjust timestamps backward from sync frame
    print("[INFO] Adjusting timestamps backward from sync frame...")
    for i in range(sync_gopro_frame - 1, -1, -1):
        gopro_timestamps_df.iloc[i, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] = (
            gopro_timestamps_df.iloc[i + 1, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] - cts_diff
        )

    # Adjust timestamps forward from sync frame
    print("[INFO] Adjusting timestamps forward from sync frame...")
    for i in range(sync_gopro_frame + 1, len(gopro_timestamps_df)):
        gopro_timestamps_df.iloc[i, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] = (
            gopro_timestamps_df.iloc[i - 1, gopro_timestamps_df.columns.get_loc('recalculated_epoch')] + cts_diff
        )

    # Ensure recalculated_epoch is in int64 (nanoseconds)
    gopro_timestamps_df['recalculated_epoch'] = gopro_timestamps_df['recalculated_epoch'].astype('int64')

    # Save the updated timestamps back to the CSV
    gopro_timestamps_df.to_csv(gopro_timestamp_file, index=False)
    print(f"[SUCCESS] Recalculated timestamps saved to {gopro_timestamp_file}")

if __name__ == "__main__":
    # Read command line arguments
    if len(sys.argv) < 3:
        exit("[ERROR] Usage: python goPro_timestamp_adjuster.py <root_dir> <sync_date>")

    base_path = sys.argv[1]
    sync_day = sys.argv[2]

    sync_offset_folder = os.path.join(base_path , 'sync_offsets', sync_day)

    sync_file_name = "sync_offset_data.csv"
    # Load sync offset data
    offset_file_path = os.path.join(sync_offset_folder, sync_file_name)

    
    # Check if the offset file exists
    if not os.path.exists(offset_file_path):
        exit(f"[ERROR] Offset file not found at {offset_file_path}")

    # Load the sync offset data
    offset_file = pd.read_csv(offset_file_path)

    # Iterate over all entries in the offset file
    for index, row in offset_file.iterrows():
        print("\n" + "="*50)
        print(f"[INFO] Processing GoPro recording: {row['gopro_file_id']}")

        gopro_timestamp_file = row['gopro_timestamp_path']
        kinect_timestamp_file = row['kinect_timestamp_path']
        sync_offset = row['sync_offset_gp-kinect']
        sync_gopro_frame = row['sync_gopro_frame']
        sync_kinect_frame = row['sync_kinect_frame']
        sync_gopro_time = row['sync_gopro_time']
        sync_kinect_time = row['sync_kinect_time']

        # update the path to the gopro timestamp file by appending _30fps.csv
        gopro_timestamp_file = gopro_timestamp_file.replace(".csv", "_30fps.csv")
        print("[INFO] GoPro timestamp file:", gopro_timestamp_file)


        # Check if both GoPro and Kinect timestamp files exist
        if not os.path.exists(gopro_timestamp_file) or not os.path.exists(kinect_timestamp_file):
            exit(f"[ERROR] Timestamps folder not found at {gopro_timestamp_file} or {kinect_timestamp_file}")

        print(f"[INFO] GoPro sync frame: {sync_gopro_frame}")
        print(f"[INFO] Kinect sync frame: {sync_kinect_frame}")
        print(f"[INFO] GoPro sync time: {sync_gopro_time}")
        print(f"[INFO] Kinect sync time: {sync_kinect_time}")
        print(f"[INFO] Sync offset: {sync_offset}")

        # Recalculate timestamps
        recalculate_timestamps_cts(gopro_timestamp_file, kinect_timestamp_file, sync_gopro_frame, sync_kinect_frame, sync_gopro_time, sync_kinect_time, sync_offset)

        print(f"[SUCCESS] Timestamps recalculated for GoPro recording: {row['gopro_file_id']}")
        print("="*50)
