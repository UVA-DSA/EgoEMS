import os
import sys
import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import cv2

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Tools.depth_sensor_processing import tools as depth_tools
import extract_depth

# Constants
CANON_K = np.array([[615.873673811006, 0, 640.803032851225], 
                    [0, 615.918359977960, 365.547839233105], 
                    [0, 0, 1]])
DEBUG = False
FRAME_RATE = 30

# Paths
GT_PATH = '/home/cogems_nist/Documents/Final/Final'
DATA_PATH = '/home/cogems_nist/Documents/Kinect_CPR_Clips/Kinect_CPR_Clips/Final/exo_kinect_cpr_clips'
LOG_PATH = './cpr_depth_quality.csv'


# Logging helpers
def init_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    else:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_log_line(format_log_line('Subject', 'Trial', 'GT Depth', 'Est Depth', 'GT Cycles', 'Est Cycles', 'Depth Error', 'Frequency Error'))

def format_log_line(sbj, t, gt_depth, est_depth, gt_cycles, est_cycles, depth_err, freq_err):
    return f"{sbj}, {t}, {gt_depth}, {est_depth}, {gt_cycles}, {est_cycles}, {depth_err}, {freq_err}"

def write_log_line(msg):
    with open(LOG_PATH, 'a') as file:
        file.write(msg + '\n')
        

# Outlier detection and filtering helpers
def clean_wrist(wrist, threshold=2.0):

    for axis in wrist:
        points = wrist[axis]
        outliers = detect_outliers_modified_z_score(points, threshold)
        points[outliers] = np.nan
        not_nan_indices = np.where(~np.isnan(points))[0]

        if len(not_nan_indices) > 0:
            points[np.isnan(points)] = np.interp(np.where(np.isnan(points))[0], not_nan_indices, points[not_nan_indices])
        else:
            print(f"Warning: No valid values in wrist {axis} to interpolate.")

        wrist[axis] = points

    return wrist

def detect_outliers_mad(data, threshold=2.0):
    median = np.median(data)
    mad = np.median([abs(x - median) for x in data])
    if mad == 0:
        print('Warning: All data entries are the same. MAD is zero.')
        return []
    return [i for i, x in enumerate(data) if abs(x - median) / mad > threshold]

def detect_outliers_modified_z_score(data, threshold=2.0):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        print('Warning: All data entries are the same. MAD is zero.')
        return []
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z_scores) > threshold)[0]


# Open3D helpers
def get_XYZ(x, y, depth, k):
    X = (x - k[0, 2]) * depth / k[0, 0]
    Y = (y - k[1, 2]) * depth / k[1, 1]
    Z = depth
    return X, Y, Z

def safe_access_depth_image(depth_img, x, y):
    if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
        return int(depth_img[y, x])
    else:
        print(f'Warning: Index out of bounds for depth image at ({x}, {y})')
        return None

def convert_p_v_to_XYZ(p_indices, v_indices, wrist_x, wrist_y, depth_imgs):
    peakXYZ_p, peakXYZ_v = [], []

    for idx in p_indices:
        depth_img_p = depth_imgs[idx]
        d_p_value = safe_access_depth_image(depth_img_p, int(wrist_x[idx]), int(wrist_y[idx]))
        if d_p_value is None:
            continue
        X_p, Y_p, Z_p = get_XYZ(int(wrist_x[idx]), int(wrist_y[idx]), d_p_value, CANON_K)
        peakXYZ_p.append([X_p, Y_p, Z_p])

    for idx in v_indices:
        depth_img_v = depth_imgs[idx]
        d_v_value = safe_access_depth_image(depth_img_v, int(wrist_x[idx]), int(wrist_y[idx]))
        if d_v_value is None:
            continue
        X_v, Y_v, Z_v = get_XYZ(int(wrist_x[idx]), int(wrist_y[idx]), d_v_value, CANON_K)
        peakXYZ_v.append([X_v, Y_v, Z_v])
    
    return peakXYZ_p, peakXYZ_v

def calculate_cpr_depth(peakXYZ_p, peakXYZ_v, DEBUG=True):
    l_min_len_peaks_valleys = min(len(peakXYZ_p), len(peakXYZ_v))
    dist_list = np.zeros(l_min_len_peaks_valleys)
    
    # Calculate distances between corresponding peaks and valleys to estimate CPR depth.
    for idx in range(l_min_len_peaks_valleys):
        p_np_array_peak = np.array(peakXYZ_p[idx])
        v_np_array_valley = np.array(peakXYZ_v[idx])
        dist = np.linalg.norm(p_np_array_peak - v_np_array_valley)
        dist_list[idx] = dist

    if DEBUG:
        plt.plot(dist_list)
        plt.title('CPR Depth')
        plt.show()
        plt.clf()

    # Filter outliers in the distance list
    dist_outliers = detect_outliers_modified_z_score(dist_list)
    dist_list[dist_outliers] = np.nan
    not_nan_dist_indices = np.where(~np.isnan(dist_list))[0]

    if len(not_nan_dist_indices) > 0:
        dist_list[np.isnan(dist_list)] = np.interp(np.where(np.isnan(dist_list))[0], not_nan_dist_indices, dist_list[not_nan_dist_indices])
    else:
        print("Warning: No valid values in dist_list to interpolate.")
    
    if DEBUG:
        plt.plot(dist_list)
        plt.title('CPR Depth Filtered')
        plt.show()
        plt.clf()
                    
    return float(np.mean(dist_list))


# File loading helpers
def read_gt_data(file_path):
    try:
        with open(file_path, 'r') as file:
            gt_lines = file.readlines()        
        gt_values = np.array([float(line.strip()) for line in gt_lines[1:]])
        return gt_values
    except FileNotFoundError:
        print(f'Error: File not found {file_path}')
        return None
    except ValueError:
        print(f'Error: Non-integer value found in {file_path}')
        return None
    
def parse_json_data(json_data) -> Dict:
    keys = sorted([int(k) for k in json_data.keys()])
    wrists = dict()

    for k in keys:
        kpt_data = json_data[str(k)]
        if 'hands' in kpt_data:
            kpt_data = kpt_data['hands']
        else:
            kpt_data = [kpt_data]
        
        if len(kpt_data) == 0: 
            continue

        for idx, hand in enumerate(kpt_data):
            if len(hand['x']) == 0:
                continue

            if idx not in wrists:
                wrists[idx] = {'x': [], 'y': [], 'frame': []}

            wrists[idx]['x'].append(hand['x'][0])
            wrists[idx]['y'].append(hand['y'][0])
            wrists[idx]['frame'].append(k)
    
    for wrist in wrists:
        wrists[wrist]['x'] = np.array(wrists[wrist]['x'], dtype=int)
        wrists[wrist]['y'] = np.array(wrists[wrist]['y'], dtype=int)
        wrists[wrist]['frame'] = np.array([wrists[wrist]['frame'][0], wrists[wrist]['frame'][-1]])

    return wrists


if __name__ == "__main__":
    init_log()

    for n in ['train_root', 'test_root', 'val_root']:
        data_dir = os.path.join(DATA_PATH, n, 'chest_compressions')
        json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
        mkv_files = [file for file in os.listdir(data_dir) if file.endswith('.mkv')]

        # Main processing loop
        for json_file in json_files:
            print(f"Processing: {json_file}")

            json_path = os.path.join(data_dir, json_file)
            with open(json_path, 'r') as file:
                json_data = json.load(file)
            
            # Match MKV file with JSON file
            mkv_file = next((f for f in mkv_files if json_file.replace('_keypoints.json', '') in f), None)

            if mkv_file is None:
                print(f'Warning: No matching MKV file found for JSON file {json_file}.')
                continue
            
            # Read depth and RGB images from video
            rgb_imgs, depth_imgs = extract_depth.read_video(os.path.join(data_dir, mkv_file))

            if len(rgb_imgs) != len(json_data.keys()):
                print(f'Warning: Number of frames in video ({len(rgb_imgs)}) does not match number in keypoints.json ({len(json_data.keys())})')
                continue

            wrists = parse_json_data(json_data)
            n_cpr_cycles_list = np.zeros(len(wrists.keys()))
            cpr_depth_list = np.zeros(len(wrists.keys()))
            
            for ind, wrist in enumerate(sorted(wrists)):
                wrist_x = wrists[wrist]['x']
                wrist_y = wrists[wrist]['y']

                # Detect peaks and valleys
                p_indices, v_indices = depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y), show=DEBUG)
                peakXYZ_p, peakXYZ_v  = convert_p_v_to_XYZ(p_indices, v_indices, wrist_x, wrist_y, depth_imgs)

                # Calculate the mean CPR depth from the distances between peaks and valleys
                mean_depth = calculate_cpr_depth(peakXYZ_p, peakXYZ_v)

                # Save data
                n_cpr_cycles_list[ind] = (len(p_indices) + len(v_indices)) * 0.5
                cpr_depth_list[ind] = mean_depth

            # Choose wrist
            wrist_ind = np.argmin(cpr_depth_list)  # Find index of minimum depth
            depth_inds = np.where((cpr_depth_list > 5) & (cpr_depth_list < 125))  # Use & for elementwise logical AND

            if depth_inds[0].size > 0:  # Check if there are any valid indices
                wrist_ind = np.random.choice(depth_inds[0])  # Use depth_inds[0] to get the actual indices

            # Get corresponding values from lists
            cpr_depth = cpr_depth_list[wrist_ind]
            n_cpr_cycles = n_cpr_cycles_list[wrist_ind]
            wrist_start_offset = int(wrists[wrist_ind]['frame'][0])
            wrist_end_offset = int(wrists[wrist_ind]['frame'][1])
            
            # Get ground truth (GT) CPR data
            sbj = json_file.split('_')[0]
            trial = json_file.split('_')[1][1:]
            base = int(float(json_file.split('_')[3]) * FRAME_RATE)
            start = base + wrist_start_offset
            end = start + wrist_end_offset

            gt_path = os.path.join(GT_PATH, sbj, 'cardiac_arrest', trial, 'distance_sensor_data', 'sync_depth_sensor.csv')
            gt_lines = read_gt_data(gt_path)

            if gt_lines is None:
                print(f"Warning: Could not read ground truth data for subject {sbj}, trial {trial}.")
                continue  # Skip this file if ground truth data could not be read

            if DEBUG:
                plt.plot(gt_lines[start:end])
                plt.title("Ground Truth Depth Data")
                plt.show()
                plt.clf()
                
            if end > len(gt_lines):
                print(f"Warning: End index {end} out of bounds for ground truth data, defaulting to first {len(rgb_imgs)} datapoints")
                start = 0
                end = len(rgb_imgs)
            
            gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_lines[start:end], show=DEBUG)
            gt_n_cpr_cycles = (len(gt_peaks) + len(gt_valleys)) * 0.5
            peak_depths_gt = gt_lines[start:end][gt_peaks]
            valley_depths_gt = gt_lines[start:end][gt_valleys]

            l_min_len_gt_peaks_valleys = min(len(peak_depths_gt), len(valley_depths_gt))
            gt_cpr_depth = float((peak_depths_gt[:l_min_len_gt_peaks_valleys] - valley_depths_gt[:l_min_len_gt_peaks_valleys]).mean())


            # Calculate time in seconds (assuming 30 frames per second)
            time_in_seconds = len(gt_lines) / FRAME_RATE

            # Calculate errors in CPR depth and frequency
            if DEBUG:
                print(f"GT: {gt_cpr_depth:.2f} mm | {gt_n_cpr_cycles:.2f} cycles")
                print(f"Est: {cpr_depth:.2f} mm | {n_cpr_cycles:.2f} cycles")

            depth_error_mm = abs(gt_cpr_depth - cpr_depth)
            n_cpr_error_per_minute = abs(gt_n_cpr_cycles - n_cpr_cycles) / time_in_seconds * 60

            # Log results
            print(f"Subject: {sbj} Trial: {trial} | CPR depth error: {depth_error_mm:.2f} mm | CPR frequency error: {n_cpr_error_per_minute:.2f} /min")
            write_log_line(format_log_line(sbj, trial, gt_cpr_depth, cpr_depth, gt_n_cpr_cycles, n_cpr_cycles, depth_error_mm, n_cpr_error_per_minute))
            
            if DEBUG:
                plt.imshow(rgb_imgs[p_indices[0]])
                plt.scatter(int(wrist_x[p_indices[0]]), int(wrist_y[p_indices[0]]), color='red')
                plt.title(f'Depth Image at Peak {p_indices[0]}')
                plt.show()
                plt.clf() 
    