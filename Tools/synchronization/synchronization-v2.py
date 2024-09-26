import os
import sys
import pandas as pd
import numpy as np
import glob

def get_depth_camera_timestamps(file_path):
    """Read depth camera timestamps from a text file."""
    # print(f"Reading depth camera timestamps from {file_path}")
    with open(file_path, 'r') as depth_txt:
        return np.array([int(line.strip()) for line in depth_txt.readlines()], dtype=np.int64)


def get_go_pro_adjusted_timestamps(gp_ts):
    """Adjust GoPro timestamps by the provided offset."""
    # return gp_ts['epoch'].astype('int64') - offset_value # -epoch and - before
    return gp_ts['recalculated_epoch'].astype('int64') 

def process_modality_sw_depthsensor(trial_path, modalities):
    """Process modality files to find smartwatch, depth sensor, GoPro, and depth camera data."""
    sw_data, depth_data =  None, None
 
    # Define the paths with wildcards
    sw_data_path = os.path.join(trial_path, 'smartwatch_data', '*', 'sw_data.csv')
    depth_data_path = os.path.join(trial_path, 'VL6180', 'VL*.csv')
    
    # Use glob to resolve wildcard paths
    sw_files = glob.glob(sw_data_path)
    depth_files = glob.glob(depth_data_path)
    
    # Read smartwatch data if the file exists
    if sw_files:
        sw_data = pd.read_csv(sw_files[0])  # Assuming you want the first match
    else:
        print(f"No smartwatch data found in: {sw_data_path}")
    
    # Read depth sensor data if the file exists
    if depth_files:
        depth_data = pd.read_csv(depth_files[0])  # Assuming you want the first match
    else:
        print(f"No depth sensor data found in: {depth_data_path}")
               
    return sw_data, depth_data


def calculate_synchronized_frames(dc_ts, gp_adj_ts):
    """Find start and end frames for GoPro and depth camera data."""
    if len(dc_ts) == 0 or len(gp_adj_ts) == 0:
        return None, None, None, None, None

    print(f"GoPro first timestamp: {gp_adj_ts[0]}")
    print(f"Depth Camera first timestamp: {dc_ts[0]}")
    
    gp_sf, dc_sf, frames = 0, 0, 0
    dc_length, gp_length = len(dc_ts), len(gp_adj_ts)

    print(f"GoPro duration frames: {gp_length} and Depth Camera duration frames: {dc_length}")
    
    # Find the last frame index for the shorter duration
    last_frame = min(dc_length, gp_length)

    print(f"Kinect First timestamp: {dc_ts[0]} and GoPro First timestamp: {gp_adj_ts[0]}")
    
    if dc_ts[0] > gp_adj_ts[0]: # GoPro starts first
        print("GoPro starts first")
        for i in range(gp_length - 1):
            if gp_adj_ts[i] >= dc_ts[0]:
                gp_sf, frames = i, min(gp_length - i, last_frame)
                break
    else: # Depth Camera starts first
        print("Depth Camera starts first")
        for i in range(dc_length - 1):
            if dc_ts[i] >= gp_adj_ts[0]:
                dc_sf, frames = i, min(dc_length - i, last_frame)
                break

    # Update start frames to be the number that is the closest a multiple of 3000/101
    print(f"GoPro start frame: {gp_sf}, Depth Camera start frame: {dc_sf}, Frames: {frames}")
    
    return gp_sf, dc_sf, frames, gp_sf + frames, dc_sf + frames

def synchronize_smartwatch_depth(sw_data, depth_data, timestamps):
    """Synchronize smartwatch and depth sensor data based on GoPro timestamps."""
    sw_timestamps = sw_data['sw_epoch_ms'].astype('int64')
    ds_timestamps = depth_data.iloc[:, 0].astype('int64')

    sw_index, depth_index = 0, 0
    sw_data_list, depth_data_list = [], []

    for gopro_epoch in timestamps:
        # Synchronize smartwatch data
        while sw_index < len(sw_timestamps) - 1 and not (sw_timestamps[sw_index] * 1e6 <= gopro_epoch < sw_timestamps[sw_index + 1] * 1e6):
            sw_index += 1
        sw_data_list.append({
            'sw_value_X_Axis': sw_data['value_X_Axis'].iloc[sw_index] if sw_index < len(sw_timestamps) else 0,
            'sw_value_Y_Axis': sw_data['value_Y_Axis'].iloc[sw_index] if sw_index < len(sw_timestamps) else 0,
            'sw_value_Z_Axis': sw_data['value_Z_Axis'].iloc[sw_index] if sw_index < len(sw_timestamps) else 0,
        })

        # Synchronize depth sensor data
        while depth_index < len(ds_timestamps) - 1 and not (ds_timestamps[depth_index] <= gopro_epoch < ds_timestamps[depth_index + 1]):
            depth_index += 1
        depth_data_list.append({
            'depth_value': depth_data.iloc[depth_index, 1] if depth_index < len(ds_timestamps) else 0
        })

    return pd.DataFrame(sw_data_list), pd.DataFrame(depth_data_list)


def synchronize(base_dir, gopro_file_path, kinect_file_path, gopro_timestamp_path, kinect_timestamp_path, gopro_sync_metadata, kinect_sync_metadata, sync_dir):
    """Synchronize trials."""

    gopro_file_id = os.path.basename(gopro_file_path).split('.')[0]
    trial_path = gopro_file_path.split('/')
    day, person, intervention, trial = trial_path[-6], trial_path[-5], trial_path[-4], trial_path[-3]

    print(f"Synchronizing {day} {person} {intervention} {trial}")
    print(f"GoPro file ID: {gopro_file_id}")

    trial_path = os.path.join(base_dir, day, person, intervention, trial)

    print(f"Trial path: {trial_path}\n")


    # # Process modality files
    sw_data, depth_data = process_modality_sw_depthsensor(
        trial_path, os.listdir(trial_path))
    
    gp_ts = pd.read_csv(gopro_timestamp_path)
    dc_ts = get_depth_camera_timestamps(kinect_timestamp_path)
    gp_adj_ts = get_go_pro_adjusted_timestamps(gp_ts)

    # print(f"{sw_data} {depth_data} {gp_ts} {gp_adj_ts} {kinect_file_path} {gopro_file_path} {dc_ts}")

    if gp_ts is None or dc_ts is None:
        print(f"Issue with GoPro or depth camera timestamps for {trial_path}")

    # # Calculate synchronized frames
    gp_sf, dc_sf, frames, gp_ef, dc_ef = calculate_synchronized_frames(dc_ts, gp_adj_ts)
    if frames is None:
        print(f"Unable to synchronize {trial_path}")

    # Calculate synchronized timestamp corresponding to start and end frames
    gp_sf_ts = gp_sf
    
    # # Append synchronized data to DataFrames
    gopro_sync_metadata = pd.concat([gopro_sync_metadata, pd.DataFrame([{'filename': gopro_file_path, 'start_frame': gp_sf, 'end_frame': gp_ef}])], ignore_index=True)
    kinect_sync_metadata = pd.concat([kinect_sync_metadata, pd.DataFrame([{'filename': kinect_file_path, 'start_frame': dc_sf, 'end_frame': dc_ef}])], ignore_index=True)

    # Synchronize smartwatch and depth sensor data
    if sw_data is not None and depth_data is not None:
        timestamps = gp_adj_ts[gp_sf:gp_ef]
        df_sw, df_ds = synchronize_smartwatch_depth(sw_data, depth_data, timestamps)

        # Write smartwatch and depth sensor data to CSVs
        for df, sensor, folder in [(df_sw, 'smartwatch', 'smartwatch_data'), (df_ds, 'depthSensor', 'depthSensor_data')]:
            output_path = os.path.join(sync_dir, day, person, intervention, trial, folder)
            os.makedirs(output_path, exist_ok=True)
            df.to_csv(os.path.join(output_path, f"{day}_{person}_{intervention}_{trial}_{sensor}.csv"), index=False)

    # print("*" * 50)

    # gopro_sync_metadata.to_json(f"{raw_data_path}/goPro_clip.json")
    # kinect_sync_metadata.to_json(f"{raw_data_path}/depthCam_clip.json")

    return gopro_sync_metadata, kinect_sync_metadata


def process_recordings(base_dir,sync_offset_path, sync_dir):
    
    # load the sync offset data csv
    sync_data = pd.read_csv(sync_offset_path)

    gopro_sync_metadata, kinect_sync_metadata  = pd.DataFrame(columns=['filename', 'start_frame', 'end_frame']), pd.DataFrame(columns=['filename', 'start_frame', 'end_frame'])

    # iterate over the rows in the csv
    for index, row in sync_data.iterrows():
        print("*"*50)

        gopro_file_path = row['gopro_file_path']
        kinect_file_path = row['kinect_file_path']
        gopro_timestamp_path = row['gopro_timestamp_path']
        kinect_timestamp_path = row['kinect_timestamp_path']

        gopro_sync_metadata, kinect_sync_metadata = synchronize(base_dir,gopro_file_path, kinect_file_path, gopro_timestamp_path, kinect_timestamp_path,gopro_sync_metadata, kinect_sync_metadata, sync_dir)

        print("*"*50)
    
    # Save accumulated metadata to JSON files after processing all rows
    print(f"\n\nSaving synchronization metadata to {base_dir}")
    gopro_sync_metadata.to_json(f"{base_dir}/gopro_clip.json", orient='records', lines=True)
    kinect_sync_metadata.to_json(f"{base_dir}/depthcam_clip.json", orient='records', lines=True)
    print("Synchronization complete.")



if __name__ == "__main__":

        # get cmd line arguments
    print(sys.argv)
    
    if len(sys.argv) < 2:
        exit("Usage: python synchronization-v2.py <path_to_root_dir>")

    base_dir = sys.argv[1]

    # one folder up from the base_dir
    sync_dir = f'{os.path.dirname(base_dir)}/Synchronized'
    os.makedirs(sync_dir, exist_ok=True)

    # Load sync_offset_data.csv
    offset_file_path = f"{base_dir}/sync_offset_data.csv"

    process_recordings(base_dir,offset_file_path,sync_dir)

    # synchronize(base_dir)
