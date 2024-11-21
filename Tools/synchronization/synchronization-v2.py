import os
import sys
import pandas as pd
import numpy as np
import glob

def get_depth_camera_timestamps(file_path):
    """Read depth camera timestamps from a text file."""
    # Log reading operation
    with open(file_path, 'r') as depth_txt:
        return np.array([int(line.strip()) for line in depth_txt.readlines()], dtype=np.int64)

def get_go_pro_adjusted_timestamps(gp_ts):
    """Adjust GoPro timestamps by the provided offset."""
    return gp_ts['recalculated_epoch'].astype('int64') 

def process_modality_sw_depthsensor(trial_path, modalities):
    """Process modality files to find smartwatch, depth sensor, GoPro, and depth camera data."""
    sw_data, depth_data =  None, None

    # Define paths with wildcards
    sw_data_path = os.path.join(trial_path, 'smartwatch_data', '*', 'sw_data.csv')
    depth_data_path = os.path.join(trial_path, 'VL6180', 'VL*.csv')

    # Use glob to resolve wildcard paths
    sw_files = glob.glob(sw_data_path)
    depth_files = glob.glob(depth_data_path)

    # Read smartwatch data if the file exists
    if sw_files:
        sw_data = pd.read_csv(sw_files[0])
    else:
        print(f"[WARNING] No smartwatch data found in: {sw_data_path}")

    # Read depth sensor data if the file exists
    if depth_files:
        depth_data = pd.read_csv(depth_files[0])
    else:
        print(f"[WARNING] No depth sensor data found in: {depth_data_path}")

    return sw_data, depth_data

def calculate_synchronized_frames(dc_ts, gp_adj_ts):
    """Find start and end frames for GoPro and depth camera data."""
    if len(dc_ts) == 0 or len(gp_adj_ts) == 0:
        return None, None, None, None, None

    print(f"[INFO] GoPro first timestamp: {gp_adj_ts[0]}")
    print(f"[INFO] Depth Camera first timestamp: {dc_ts[0]}")

    dc_length, gp_length = len(dc_ts), len(gp_adj_ts)
    print(f"[INFO] GoPro duration: {gp_length} frames, Depth Camera duration: {dc_length} frames")

    last_frame = min(dc_length, gp_length)
    gp_sf, dc_sf, frames = 0, 0, 0

    # find the matching start frame for GoPro and Depth Camera

    
    if dc_ts[0] > gp_adj_ts[0]:  # GoPro starts first
        print("[INFO] GoPro starts first")
        for i in range(gp_length - 1):
            if gp_adj_ts[i] >= dc_ts[0]:
                gp_sf, frames = i, min(gp_length - i, last_frame)
                break
    else:  # Depth Camera starts first
        print("[INFO] Depth Camera starts first")
        for i in range(dc_length - 1):
            if dc_ts[i] >= gp_adj_ts[0]:
                dc_sf, frames = i, min(dc_length - i, last_frame)
                break

    print(f"[INFO] GoPro start frame: {gp_sf}, Depth Camera start frame: {dc_sf}, Frames: {frames}")
    return gp_sf, dc_sf, frames, gp_sf + frames, dc_sf + frames

def synchronize_smartwatch_depth(sw_data, depth_data, timestamps):
    """Synchronize smartwatch and depth sensor data based on GoPro timestamps."""
    sw_timestamps = sw_data['sw_epoch_ms'].astype('int64')
    ds_timestamps = depth_data.iloc[:, 0].astype('int64') if depth_data is not None else [0]
    
    print(f"[INFO] Smartwatch timestamps: {len(sw_timestamps)}, Depth sensor timestamps: {len(ds_timestamps)}, GoPro timestamps: {len(timestamps)}")

    sw_index, depth_index = 0, 0
    sw_data_list, depth_data_list = [], []


    for gopro_epoch in timestamps:
        
        # Synchronize smartwatch data
               #print(f"gopro_epoch: {gopro_epoch}")
        found_sw_data = False
        while sw_index < len(sw_timestamps) - 1:
            if gopro_epoch >= sw_timestamps[sw_index] * 1000000 and gopro_epoch < sw_timestamps[sw_index + 1] * 1000000:
                sw_data_list.append({
                    'sw_value_X_Axis': sw_data['value_X_Axis'].iloc[sw_index],
                    'sw_value_Y_Axis': sw_data['value_Y_Axis'].iloc[sw_index],
                    'sw_value_Z_Axis': sw_data['value_Z_Axis'].iloc[sw_index],
                })
                #print("sw data found!")
                found_sw_data = True
                break
            sw_index += 1
                
        #for this timestamp, if we cannot find sw data, we should put in zeroes
        if not found_sw_data:
            #print("MISSING SW DATA")
            sw_data_list.append({
                'sw_value_X_Axis': 0,
                'sw_value_Y_Axis': 0,
                'sw_value_Z_Axis': 0,
            })
            sw_index = 0

        found_ds_data = False
        #print(f"this is gopro_epoch: {gopro_epoch} {type(gopro_epoch)} and this is ds_timestamp: {ds_timestamps[depth_index]} {type(ds_timestamps[depth_index])}")
        while depth_index < len(ds_timestamps) - 1:
            #print(depth_index)
            if np.int64(gopro_epoch) >= ds_timestamps[depth_index] and np.int64(gopro_epoch) < ds_timestamps[depth_index + 1]:
                depth_data_list.append({
                    'depth_value': depth_data[depth_data.columns[1]].iloc[depth_index],
                })
                #print("depth camera data found!")
                print("[INFO] Depth sensor data found!")
                found_ds_data = True
                break
            depth_index += 1
                
        #for this timestamp, if we cannot find ds data, we should put in zeroes
        if not found_ds_data:
            #print("missing DEPTH DATA")
            depth_data_list.append({
                    'depth_value': 0,
                }) 
            depth_index = 0

    return pd.DataFrame(sw_data_list), pd.DataFrame(depth_data_list)

def synchronize(base_dir, gopro_file_path, kinect_file_path, gopro_timestamp_path, kinect_timestamp_path, gopro_sync_metadata, kinect_sync_metadata):
    """Synchronize trials."""
    gopro_file_id = os.path.basename(gopro_file_path).split('.')[0]
    trial_path_parts = gopro_file_path.split('/')
    print(trial_path_parts)
    day, person, intervention, trial = trial_path_parts[-6], trial_path_parts[-5], trial_path_parts[-4], trial_path_parts[-3]

    if ("Lahiru" in trial_path_parts):
        intervention = ""
        person = trial_path_parts[-4]
        day = ""

    print(f"[INFO] Synchronizing {day} {person} {intervention} {trial}")
    print(f"[INFO] GoPro file ID: {gopro_file_id}")

    trial_path = os.path.join(base_dir, day, person, intervention, trial)
    print(f"[INFO] Trial path: {trial_path}\n")

    # Process modality files
    sw_data, depth_data = process_modality_sw_depthsensor(trial_path, os.listdir(trial_path))
    gp_ts = pd.read_csv(gopro_timestamp_path)
    dc_ts = get_depth_camera_timestamps(kinect_timestamp_path)
    gp_adj_ts = get_go_pro_adjusted_timestamps(gp_ts)

    if gp_ts is None or dc_ts is None:
        print(f"[ERROR] Issue with GoPro or depth camera timestamps for {trial_path}")
        return gopro_sync_metadata, kinect_sync_metadata

    gp_sf, dc_sf, frames, gp_ef, dc_ef = calculate_synchronized_frames(dc_ts, gp_adj_ts)
    if frames is None:
        print(f"[ERROR] Unable to synchronize {trial_path}")
        return gopro_sync_metadata, kinect_sync_metadata

    # Append synchronized metadata
    gopro_sync_metadata = pd.concat([gopro_sync_metadata, pd.DataFrame([{'filename': gopro_file_path, 'start_frame': gp_sf, 'end_frame': gp_ef}])], ignore_index=True)
    kinect_sync_metadata = pd.concat([kinect_sync_metadata, pd.DataFrame([{'filename': kinect_file_path, 'start_frame': dc_sf, 'end_frame': dc_ef}])], ignore_index=True)


    # Synchronize smartwatch and depth sensor data
    if sw_data is not None: # for ecg trials no need depth sensor
    # if sw_data is not None and depth_data is not None:
        timestamps = gp_adj_ts[gp_sf:gp_ef]
        df_sw, df_ds = synchronize_smartwatch_depth(sw_data, depth_data, timestamps)
        print(f"[INFO] Synchronized smartwatch data: {len(df_sw)}, depth sensor data: {len(df_ds)}")

        # update trial path
        trial_path = trial_path.replace('/standard/storage/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/23-10-2024', '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final')
        print(f"[INFO] Saving synchronized data to {trial_path}")
        sync_sw_csv_path = os.path.join(trial_path, 'smartwatch_data')
        # sync_ds_csv_path = os.path.join(trial_path, 'distance_sensor_data')

        os.makedirs(sync_sw_csv_path, exist_ok=True)
        # os.makedirs(sync_ds_csv_path, exist_ok=True)

        df_sw.to_csv(os.path.join(sync_sw_csv_path, f"sync_smartwatch.csv"), index=False)
        # df_ds.to_csv(os.path.join(sync_ds_csv_path, f"sync_depth_sensor.csv"), index=False)

    return gopro_sync_metadata, kinect_sync_metadata

def process_recordings(base_dir, sync_offset_path):
    """Process and synchronize recordings based on sync offset data."""
    sync_data = pd.read_csv(sync_offset_path)

    gopro_sync_metadata, kinect_sync_metadata = pd.DataFrame(columns=['filename', 'start_frame', 'end_frame']), pd.DataFrame(columns=['filename', 'start_frame', 'end_frame'])

    # Iterate over the rows in the sync data CSV
    for index, row in sync_data.iterrows():
        print("\n" + "="*50)
        print(f"[INFO] Processing trial at index: {index}")

        gopro_file_path = row['gopro_file_path']
        kinect_file_path = row['kinect_file_path']
        gopro_timestamp_path = row['gopro_timestamp_path']
        kinect_timestamp_path = row['kinect_timestamp_path']
        
        # remove _fps_converted from the file name of kinect file if it exists
        kinect_file_path = kinect_file_path.replace('_fps_converted', '')

        gopro_sync_metadata, kinect_sync_metadata = synchronize(base_dir, gopro_file_path, kinect_file_path, gopro_timestamp_path, kinect_timestamp_path, gopro_sync_metadata, kinect_sync_metadata)

        print("="*50 + "\n")
    
    print(f"\n[INFO] Saving synchronization metadata to {base_dir}")
    gopro_sync_metadata.to_json(f"{base_dir}/gopro_clip.json")
    kinect_sync_metadata.to_json(f"{base_dir}/depthcam_clip.json")
    print("[INFO] Synchronization complete.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit("[ERROR] Usage: python synchronization-v2.py <path_to_root_dir> <path_to_sync_offset_data_folder>")

    base_dir = sys.argv[1]
    sync_day = sys.argv[2]


    sync_offset_folder = os.path.join(base_dir , 'sync_offsets', sync_day)

    sync_file_name = "sync_offset_data.csv"

    # Load sync offset data
    offset_file_path = os.path.join(sync_offset_folder, sync_file_name)

    if not os.path.exists(offset_file_path):
        exit(f"[ERROR] Sync offset file not found: {offset_file_path}")

    print(f"[INFO] Processing recordings in {offset_file_path}")
    process_recordings(base_dir, offset_file_path)