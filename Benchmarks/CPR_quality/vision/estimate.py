import os
import sys
import json
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
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
def init_log(log_path):
    """Initialize log by removing existing file or creating directory."""
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_log_line(log_path, format_log_line('Subject', 'Trial', 'Window', 'GT Depth', 'Est Depth', 'GT Cycles', 'Est Cycles', 'Depth Error', 'Frequency Error'))

def format_log_line(sbj, t, window, gt_depth, est_depth, gt_cycles, est_cycles, depth_err, freq_err):
    """Format log line for CSV output."""
    return f"{sbj}, {t}, {window}, {gt_depth}, {est_depth}, {gt_cycles}, {est_cycles}, {depth_err}, {freq_err}"

def write_log_line(log_path, msg):
    """Append log message to the log file."""
    with open(log_path, 'a') as file:
        file.write(msg + '\n')

# File helpers
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
        wrists[wrist]['x'] = np.array(wrists[wrist]['x'], dtype=float)
        wrists[wrist]['y'] = np.array(wrists[wrist]['y'], dtype=float)
        wrists[wrist]['frame'] = np.array([wrists[wrist]['frame'][0], wrists[wrist]['frame'][-1]], dtype=float)

    return wrists

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

# Outlier detection and filtering helpers
def clean_wrist(wrist: Dict[str, np.ndarray], threshold=2.0) -> Dict[str, np.ndarray]:
    """Clean wrist data by detecting and interpolating outliers."""
    for axis in wrist:
        points = wrist[axis]
        outliers = detect_outliers_modified_z_score(points, threshold)

        if len(outliers) == 0:
            continue

        points[outliers] = np.nan
        not_nan_indices = np.where(~np.isnan(points))[0]

        if len(not_nan_indices) > 0:
            points[np.isnan(points)] = np.interp(np.where(np.isnan(points))[0], not_nan_indices, points[not_nan_indices])
        else:
            print(f"Warning: No valid values in wrist {axis} to interpolate.")

        wrist[axis] = points

    return wrist

def detect_outliers_modified_z_score(data: np.ndarray, threshold=2.0) -> np.ndarray:
    """Detect outliers using modified Z-score method."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        print('Warning: All data entries are the same. MAD is zero.')
        return np.array([])

    modified_z_scores = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z_scores) > threshold)[0]


# Open3D helpers
def get_XYZ(x: int, y: int, depth: int, k: np.ndarray) -> tuple:
    """Convert pixel coordinates and depth to 3D XYZ coordinates."""
    X = (x - k[0, 2]) * depth / k[0, 0]
    Y = (y - k[1, 2]) * depth / k[1, 1]
    Z = depth
    return X, Y, Z

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

def safe_access_depth_image(depth_img: np.ndarray, x: int, y: int) -> int:
    """Safely access a pixel value from a depth image."""
    if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
        return int(depth_img[y, x])
    else:
        print(f'Warning: Index out of bounds for depth image at ({x}, {y})')
        return None


# CPR Depth Calculation Helpers
def process_ground_truth_data(sbj: str, trial: str, start: int, end: int) -> tuple:
    gt_path = os.path.join(GT_PATH, sbj, 'cardiac_arrest', trial, 'distance_sensor_data', 'sync_depth_sensor.csv')
    gt_lines = read_gt_data(gt_path)

    if gt_lines is None:
        print(f"Warning: Could not read ground truth data for subject {sbj}, trial {trial}.")
        return None, None
        
    if end > len(gt_lines):
        print(f"Warning: End index {end} out of bounds for ground truth data, defaulting to first {end - start} datapoints")
        start = 0
        end = end - start
    
    gt_lines = gt_lines[start:end]
    gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_lines, show=DEBUG)
    gt_n_cpr_cycles = (len(gt_peaks) + len(gt_valleys)) * 0.5
    peak_depths_gt = gt_lines[gt_peaks]
    valley_depths_gt = gt_lines[gt_valleys]

    l_min_len_gt_peaks_valleys = min(len(peak_depths_gt), len(valley_depths_gt))
    gt_cpr_depth = float((peak_depths_gt[:l_min_len_gt_peaks_valleys] - valley_depths_gt[:l_min_len_gt_peaks_valleys]).mean())

    return gt_cpr_depth, gt_n_cpr_cycles

def calculate_cpr_depth(peakXYZ_p: list, peakXYZ_v: list) -> float:
    """Calculate CPR depth based on distances between peaks and valleys."""
    l_min_len_peaks_valleys = min(len(peakXYZ_p), len(peakXYZ_v))
    
    if l_min_len_peaks_valleys == 0:
        print("Warning: No valid peaks or valleys found.")
        return float('nan')

    dist_list = np.zeros(l_min_len_peaks_valleys)

    for idx in range(l_min_len_peaks_valleys):
        p_np_array_peak = np.array(peakXYZ_p[idx])
        v_np_array_valley = np.array(peakXYZ_v[idx])
        dist_list[idx] = np.linalg.norm(p_np_array_peak - v_np_array_valley)

    # Filter outliers in the distance list
    dist_outliers = detect_outliers_modified_z_score(dist_list)

    if len(dist_outliers) == 0:
        return float(np.mean(dist_list))

    dist_list[dist_outliers] = np.nan

    not_nan_dist_indices = np.where(~np.isnan(dist_list))[0]
    
    if len(not_nan_dist_indices) > 0:
        dist_list[np.isnan(dist_list)] = np.interp(np.where(np.isnan(dist_list))[0], not_nan_dist_indices, dist_list[not_nan_dist_indices])
    
    return float(np.mean(dist_list))

def choose_wrist(wrists, cpr_depth_list, cpr_freq_list):
    """Choose the wrist with the minimum depth and a minimum number of frames."""
    wrist_ind = 0  # Find index of minimum depth
    depth_inds = np.where((cpr_depth_list > 20)) 

    if depth_inds[0].size > 0:  # Check if there are any valid indices
        wrist_depth = np.min(cpr_depth_list[depth_inds])
        potential_wrist_ind =  list(cpr_depth_list).index(wrist_depth)

        if wrists[potential_wrist_ind]['frame'][1] - wrists[potential_wrist_ind]['frame'][0] > 150:
            wrist_ind = potential_wrist_ind

    return 0

def get_video_start_end(wrist: Dict[str, np.ndarray], json_file: str) -> tuple:
    """Get the start and end indices of the video frames based on the wrist data."""
    # Get corresponding values from lists
    wrist_start_offset = int(wrist['frame'][0])
    wrist_end_offset = int(wrist['frame'][1])
    
    # Get ground truth (GT) CPR data
    base = int(float(json_file.split('_')[3]) * FRAME_RATE)
    start = base + wrist_start_offset
    end = start + wrist_end_offset
    return start, end

# JSON Processing Target Function
def process_trial(json_data, rgb_imgs, depth_imgs, window):
    """Process a single trial (5 seconds of JSON Data) and return CPR depth and frequency estimates."""
    wrists = parse_json_data(json_data)
    n_cpr_cycles_list = np.zeros(len(wrists.keys()))
    cpr_depth_list = np.zeros(len(wrists.keys()))
    
    for ind, wrist in enumerate(sorted(wrists)):
        wrist_x = wrists[wrist]['x']
        wrist_y = wrists[wrist]['y']
        wrist = clean_wrist(wrists[wrist])

        # Detect peaks and valleys
        try:
            p_indices, v_indices = depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y), show=DEBUG)
        except ZeroDivisionError:
            print(f"Error: Zero division error in peak/valley detection for {json_file}.")
            continue
        peakXYZ_p, peakXYZ_v  = convert_p_v_to_XYZ(p_indices, v_indices, wrist_x, wrist_y, depth_imgs)

        # Calculate the mean CPR depth from the distances between peaks and valleys
        mean_depth = calculate_cpr_depth(peakXYZ_p, peakXYZ_v)

        # Save data
        n_cpr_cycles_list[ind] = (len(p_indices) + len(v_indices)) * 0.5
        cpr_depth_list[ind] = mean_depth
    
    # Choose wrist
    wrist_ind = choose_wrist(wrists, cpr_depth_list, n_cpr_cycles_list)
    cpr_depth, n_cpr_cycles = cpr_depth_list[wrist_ind], n_cpr_cycles_list[wrist_ind]

    # Extract video data
    start, end = get_video_start_end(wrists[wrist_ind], json_file)
    sbj, trial = json_file.split('_')[0], json_file.split('_')[1][1:]

    # Process ground truth data
    gt_cpr_depth, gt_n_cpr_cycles = process_ground_truth_data(sbj, trial, start, end)
    
    if gt_cpr_depth is None or gt_n_cpr_cycles is None:
        return None

    # Calculate errors
    time_in_seconds = (end - start) / FRAME_RATE
    depth_error_mm = abs(gt_cpr_depth - cpr_depth)
    n_cpr_error_per_minute = abs(gt_n_cpr_cycles - n_cpr_cycles) / time_in_seconds * 60

    # Return results
    print(f"Subject: {sbj} Trial: {trial} | CPR depth error: {depth_error_mm:.2f} mm | CPR frequency error: {n_cpr_error_per_minute:.2f} /min")
    return format_log_line(sbj, trial, window, gt_cpr_depth, cpr_depth, gt_n_cpr_cycles, n_cpr_cycles, depth_error_mm, n_cpr_error_per_minute)


# Main Execution Block
if __name__ == "__main__":
    # Initialize logging
    init_log(LOG_PATH)

    print(f"Debug mode is {DEBUG}.")

    # Iterate over each dataset folder (train/test/val)
    for n in ['val_root', 'train_root', 'test_root']:
        
        data_dir = os.path.join(DATA_PATH, n, 'chest_compressions')
        
        json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
        mkv_files = [file for file in os.listdir(data_dir) if file.endswith('.mkv')]
        results = []

        # Process each JSON file (trial)
        for json_file in json_files:
            json_path = os.path.join(data_dir, json_file)
            with open(json_path, 'r') as file:
                json_data = json.load(file)

            # Read JSON data
            json_path = os.path.join(data_dir, json_file)
            with open(json_path, 'r') as file:
                json_data = json.load(file)

            # Match MKV file with JSON file
            mkv_file = next((f for f in mkv_files if json_file.replace('_keypoints.json', '') in f), None)

            if mkv_file is None:
                print(f'Warning: No matching MKV file found for JSON file {json_file}.')
                continue

            # Read video frames
            rgb_imgs, depth_imgs = extract_depth.read_video(os.path.join(data_dir, mkv_file))

            if len(rgb_imgs) != len(json_data.keys()):
                print(f'Warning: Number of frames in video ({len(rgb_imgs)}) does not match number in keypoints.json ({len(json_data.keys())})')
                continue

            for start in range(0, len(rgb_imgs), FRAME_RATE * 5):
                end = start + FRAME_RATE * 5
                json_data_window = {str(k): json_data[str(k)] for k in range(start, end) if str(k) in json_data}

                if not json_data_window:
                    break

                result = process_trial(json_data_window, rgb_imgs[start:end], depth_imgs[start:end], start // (FRAME_RATE * 5))

                if result is not None:
                    write_log_line(LOG_PATH, result)