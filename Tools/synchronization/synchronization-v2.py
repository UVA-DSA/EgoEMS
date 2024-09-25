import os
import sys
import pandas as pd
import numpy as np

raw_data_path = "/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
goPro_timestamps = f"{raw_data_path}/goPro_timestamps"

offsets = pd.read_csv(f"{raw_data_path}/offsets.csv")


def find_matching_offset(day):
    """Find the offset for the given day from the offsets CSV."""
    matching_offset = offsets[offsets['day'] == day]
    if not matching_offset.empty:
        return matching_offset.iloc[0, 1]
    return None


def get_depth_camera_timestamps(file_path):
    """Read depth camera timestamps from a text file."""
    with open(file_path, 'r') as depth_txt:
        return np.array([int(line.strip()) for line in depth_txt.readlines()], dtype=np.int64)


def get_go_pro_adjusted_timestamps(gp_ts, offset_value):
    """Adjust GoPro timestamps by the provided offset."""
    # return gp_ts['epoch'].astype('int64') - offset_value # -epoch and - before
    return gp_ts['recalculated_epoch'].astype('int64') - offset_value # -epoch and - before


def process_modality_files(trial_path, modalities):
    """Process modality files to find smartwatch, depth sensor, GoPro, and depth camera data."""
    sw_data, depth_data, gp_ts, gp_adj_ts = None, None, None, None
    dc_video_path, gp_video_path, dc_ts = None, None, None

    for modality in modalities:
        modality_path = os.path.join(trial_path, modality)
        for file in os.listdir(modality_path):
            file_path = os.path.join(modality_path, file)
            
            if file.startswith("VL"):  # Depth sensor data CSV
                depth_data = pd.read_csv(file_path)

            elif file.endswith(".csv"):  # Smartwatch folder
                sw_data = pd.read_csv(os.path.join(modality_path, file))

            elif file.endswith(".mkv"):  # Depth camera video
                dc_video_path = file_path

            elif file.endswith(".txt"):  # Depth camera timestamps
                dc_ts = get_depth_camera_timestamps(file_path)

            elif file.endswith(".MP4"):  # GoPro video
                gp_video_path = file_path
                for csv in os.listdir(goPro_timestamps):
                    if os.path.splitext(file)[0] == os.path.splitext(csv)[0]:  # Match GoPro timestamp file
                        gp_ts = pd.read_csv(os.path.join(goPro_timestamps, csv))
                        offset_value = find_matching_offset(day)
                        print(f"Offset value {offset_value} for GoPro: {os.path.splitext(file)[0]} ")
                        if offset_value:
                            gp_adj_ts = get_go_pro_adjusted_timestamps(gp_ts, offset_value)
                            print(f"GoPro timestamp adjusted  {gp_ts} for {os.path.splitext(file)[0]}")
                            
                            # add adjusted offset timestamp column to the GoPro timestamp file
                            gp_ts['epoch_adj'] = gp_adj_ts
                            
                            # Overwrite adjusted GoPro timestamps to CSV
                            gp_ts.to_csv(os.path.join(goPro_timestamps, csv), index=False)
                            
    return sw_data, depth_data, gp_ts, gp_adj_ts, dc_video_path, gp_video_path, dc_ts


def calculate_synchronized_frames(dc_ts, gp_adj_ts):
    """Find start and end frames for GoPro and depth camera data."""
    if len(dc_ts) == 0 or len(gp_adj_ts) == 0:
        return None, None, None, None, None

    
    
    gp_sf, dc_sf, frames = 0, 0, 0
    dc_length, gp_length = len(dc_ts), len(gp_adj_ts)

    print(f"GoPro duration frames: {gp_length} and Depth Camera duration frames: {gp_length}")
    
    # Find the last frame index for the shorter duration
    last_frame = min(dc_length, gp_length)
    
    if dc_ts[0] > gp_adj_ts[0]: # GoPro starts first
        for i in range(gp_length - 1): 
            # if gp_adj_ts[i] <= dc_ts[0] < gp_adj_ts[i + 1]:
            if gp_adj_ts[i] >= dc_ts[0]:
                print(f"GoPro start frame: {gp_adj_ts[i]}")
                dc_sf, frames = i, min(gp_length - i, dc_length)
                break
    else: # Depth Camera starts first
        for i in range(dc_length - 1):
            if dc_ts[i] >= gp_adj_ts[0]:
                print(f"GoPro start frame: {gp_adj_ts[i]}")
                gp_sf, frames = i, min(dc_length - i, gp_length)
                break

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


def synchronize(day):
    goPro_json = pd.DataFrame(columns=['filename', 'start_frame', 'end_frame'])
    dS_json = pd.DataFrame(columns=['filename', 'start_frame', 'end_frame'])

    for person in os.listdir(day):
        person_path = os.path.join(day, person)
        # if(person != "bryan"):
        #     continue
        for intervention in os.listdir(person_path):
            intervention_path = os.path.join(person_path, intervention)
            for trial in os.listdir(intervention_path):
                # if trial != "0":
                #     continue
                trial_path = os.path.join(intervention_path, trial)

                # Process modality files
                sw_data, depth_data, gp_ts, gp_adj_ts, dc_video_path, gp_video_path, dc_ts = process_modality_files(
                    trial_path, os.listdir(trial_path))

                if gp_ts is None or dc_ts is None:
                    print(f"Issue with GoPro or depth camera timestamps for {trial_path}")
                    continue

                # Calculate synchronized frames
                gp_sf, dc_sf, frames, gp_ef, dc_ef = calculate_synchronized_frames(dc_ts, gp_adj_ts)
                if frames is None:
                    continue

                # Calculate synchronized timestamp corresponding to start and end frames
                gp_sf_ts = gp_sf
                
                # Append synchronized data to DataFrames
                goPro_json = pd.concat([goPro_json, pd.DataFrame([{'filename': gp_video_path, 'start_frame': gp_sf, 'end_frame': gp_ef}])], ignore_index=True)
                dS_json = pd.concat([dS_json, pd.DataFrame([{'filename': dc_video_path, 'start_frame': dc_sf, 'end_frame': dc_ef}])], ignore_index=True)

                                # Synchronize smartwatch and depth sensor data
                if sw_data is not None and depth_data is not None:
                    timestamps = gp_adj_ts[gp_sf:gp_ef]
                    df_sw, df_ds = synchronize_smartwatch_depth(sw_data, depth_data, timestamps)

                    # Write smartwatch and depth sensor data to CSVs
                    for df, sensor, folder in [(df_sw, 'smartwatch', 'smartwatch_data'), (df_ds, 'depthSensor', 'depthSensor_data')]:
                        output_path = os.path.join("synchronized", day, person, intervention, trial, folder)
                        os.makedirs(output_path, exist_ok=True)
                        df.to_csv(os.path.join(output_path, f"{day}_{person}_{intervention}_{trial}_{sensor}.csv"), index=False)

    goPro_json.to_json(f"{raw_data_path}/goPro_clip.json")
    dS_json.to_json(f"{raw_data_path}/depthCam_clip.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        day = sys.argv[1]
        path = f"{raw_data_path}/{day}"
        synchronize(path)
    else:
        print("No folder provided")
