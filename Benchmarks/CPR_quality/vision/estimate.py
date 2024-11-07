import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from Tools.depth_sensor_processing import tools as depth_tools
import extract_depth

CANON_K = np.array([[615.873673811006, 0, 640.803032851225], 
                    [0, 615.918359977960, 365.547839233105], 
                    [0, 0, 1]])

def get_XYZ(x, y, depth, k):
    X = (x - k[0, 2]) * depth / k[0, 0]
    Y = (y - k[1, 2]) * depth / k[1, 1]
    Z = depth
    return X, Y, Z

def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')

def detect_outliers_mad(data, threshold=3):
    median = np.median(data)
    mad = np.median([abs(x - median) for x in data])
    if mad == 0:
        print('Warning: All data entries are the same. MAD is zero.')
        return []
    return [i for i, x in enumerate(data) if abs(x - median) / mad > threshold]

def safe_access_depth_image(depth_img, x, y):
    if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
        return int(depth_img[y, x])
    else:
        print(f'Warning: Index out of bounds for depth image at ({x}, {y})')
        return None

def read_gt_data(file_path):
    try:
        with open(file_path, 'r') as file:
            gt_lines = file.readlines()
        gt_values = np.array([int(line.strip()) for line in gt_lines[1:]])
        return gt_values
    except FileNotFoundError:
        print(f'Error: File not found {file_path}')
        return None
    except ValueError:
        print(f'Error: Non-integer value found in {file_path}')
        return None

GT_path = '/home/cogems_nist/Documents/Final/Final'
data_path = '/home/cogems_nist/Documents/Kinect_CPR_Clips/Kinect_CPR_Clips/exo_kinect_cpr_clips'
log_path = './cpr_depth_quality.txt'
DEBUG = True

init_log(log_path)

for n in ['train_root', 'test_root', 'val_root']:
    data_dir = os.path.join(data_path, n, 'chest_compressions')
    
    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
    mkv_files = [file for file in os.listdir(data_dir) if file.endswith('.mkv')]

    for json_file in json_files:
        print(f"Processing: {json_file}")
        json_path = os.path.join(data_dir, json_file)
        json_data = json.load(open(json_path))
        
        # Match MKV file with JSON file
        mkv_file = next((f for f in mkv_files if json_file.replace('_keypoints.json', '') in f), None)
        if mkv_file is None:
            continue
        
        # Read depth and RGB images from video
        rgb_imgs, depth_imgs = extract_depth.read_video(os.path.join(data_dir, mkv_file))
        frames = len(rgb_imgs)

        if len(rgb_imgs) != len(json_data.keys()):
            print(f'Warning: Number of frames in video ({len(rgb_imgs)}) does not match number in keypoints.json ({len(json_data.keys())})')
            continue
        
        keys = sorted([int(k) for k in json_data.keys()])
        
        wrist_x, wrist_y = [], []
        key_list = []

        for k in keys:
            kpt_data = json_data[str(k)]
            if len(kpt_data['x']) == 0:
                continue
            wrist_x.append(kpt_data['x'][0])
            wrist_y.append(kpt_data['y'][0])
            key_list.append(k)

        wrist_x = np.array(wrist_x, dtype=float)
        wrist_y = np.array(wrist_y, dtype=float)

        if DEBUG:
            # Plot Wrist X
            plt.plot(wrist_x)
            plt.title('Wrist X')
            plt.show()  
            plt.clf()
            
        """
        # Normalize and filter outliers based on squared values
        vals_y = wrist_y**2
        vals_y =(vals_y - vals_y.min()) / (vals_y.max() - vals_y.min())
        outlier_idx_y = np.where(vals_y > 0.3)[0]
        wrist_y[outlier_idx_y] = np.nan
        not_nan_y_indices = np.where(~np.isnan(wrist_y))[0]

        vals_x = wrist_x**2
        vals_x = (vals_x - vals_x.min()) / (vals_x.max() - vals_x.min())
        outlier_idx_x = np.where(vals_x > 0.3)[0]
        wrist_x[outlier_idx_x] = np.nan
        not_nan_x_indices = np.where(~np.isnan(wrist_x))[0]

        if len(not_nan_y_indices) == 0 or len(not_nan_x_indices) == 0:
            print(wrist_x)
            print(wrist_y)
            print(f"Warning: No valid wrist position data available for {json_file}.")
            continue

        # Interpolate
        wrist_y[outlier_idx_y] = np.interp(outlier_idx_y, not_nan_y_indices, wrist_y[not_nan_y_indices])
        wrist_x[outlier_idx_x] = np.interp(outlier_idx_x, not_nan_x_indices, wrist_x[not_nan_x_indices])
        """

        # Normalize without squaring
        vals_y = (wrist_y - wrist_y.min()) / (wrist_y.max() - wrist_y.min())
        vals_x = (wrist_x - wrist_x.min()) / (wrist_x.max() - wrist_x.min())

        outlier_idx_y = np.where(vals_y > 0.9)[0]  # Adjust threshold as needed
        outlier_idx_x = np.where(vals_x > 0.9)[0]

        wrist_y[outlier_idx_y] = np.nan
        wrist_x[outlier_idx_x] = np.nan

        # Interpolate missing values (NaNs) for wrist_y
        not_nan_y_indices = np.where(~np.isnan(wrist_y))[0]
        if len(not_nan_y_indices) > 0:  # Check if there are any valid indices
            wrist_y[np.isnan(wrist_y)] = np.interp(np.where(np.isnan(wrist_y))[0], not_nan_y_indices, wrist_y[not_nan_y_indices])
        else:
            print("Warning: No valid values in wrist_y to interpolate.")
            continue  # Skip further processing for this file

        # Interpolate missing values (NaNs) for wrist_x
        not_nan_x_indices = np.where(~np.isnan(wrist_x))[0]
        if len(not_nan_x_indices) > 0:  # Check if there are any valid indices
            wrist_x[np.isnan(wrist_x)] = np.interp(np.where(np.isnan(wrist_x))[0], not_nan_x_indices, wrist_x[not_nan_x_indices])
        else:
            print("Warning: No valid values in wrist_x to interpolate.")
            continue  # Skip further processing for this file

        if DEBUG:
            # Plot filtered Wrist X
            plt.plot(wrist_x)
            plt.title('Wrist X (Filtered)')
            plt.show() 
            plt.clf()  

        # Detect peaks and valleys
        p_indices, v_indices = depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y), mul=1, show=DEBUG)

        n_cpr_cycles = (len(p_indices) + len(v_indices)) * 0.5

        # Process depth images at peaks and valleys
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

        l_min_len_peaks_valleys = min(len(peakXYZ_p), len(peakXYZ_v))

        dist_list_cpr_depths_meaningful_diffs_between_peaks_valleys=[]
        
        # Calculate distances between corresponding peaks and valleys to estimate CPR depth.
        for idx in range(l_min_len_peaks_valleys):
             p_np_array_peak = np.array(peakXYZ_p[idx])
             v_np_array_valley = np.array(peakXYZ_v[idx])
             dist = np.sum(np.linalg.norm(p_np_array_peak - v_np_array_valley))
             dist_list_cpr_depths_meaningful_diffs_between_peaks_valleys.append(float(dist))

        # Calculate the mean CPR depth from the distances between peaks and valleys
        cpr_depth = float(np.mean(dist_list_cpr_depths_meaningful_diffs_between_peaks_valleys))

        # Get ground truth (GT) CPR data
        sbj = json_file.split('_')[0]
        t = json_file.split('_')[1][1:]

        gt_path = os.path.join(GT_path, sbj, 'cardiac_arrest', t, 'distance_sensor_data', 'sync_depth_sensor.csv')
        gt_lines = read_gt_data(gt_path)

        if gt_lines is None:
            print(f"Warning: Could not read ground truth data for subject {sbj}, trial {t}.")
            continue  # Skip this file if ground truth data could not be read
        
        gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_lines[:frames], mul=5, show=DEBUG)
        gt_n_cpr_cycles = (len(gt_peaks) + len(gt_valleys)) * 0.5

        peak_depths_gt = gt_lines[gt_peaks]
        valley_depths_gt = gt_lines[gt_valleys]

        l_min_len_gt_peaks_valleys = min(len(peak_depths_gt), len(valley_depths_gt))
        gt_cpr_depth = float((peak_depths_gt[:l_min_len_gt_peaks_valleys] - valley_depths_gt[:l_min_len_gt_peaks_valleys]).mean())

        # Calculate time in seconds (assuming 30 frames per second)
        time_in_seconds = len(gt_lines) / 30

        # Calculate errors in CPR depth and frequency
        if DEBUG:
            print(f"GT: {gt_cpr_depth:.2f} mm | {gt_n_cpr_cycles:.2f} cycles")
            print(f"Est: {cpr_depth:.2f} mm | {n_cpr_cycles:.2f} cycles")

        depth_error_mm = abs(gt_cpr_depth - cpr_depth)
        n_cpr_error_per_minute = abs(gt_n_cpr_cycles - n_cpr_cycles) / time_in_seconds * 60

        # Log results
        msg = f'Subject: {sbj} Trial: {t} | CPR depth error: {depth_error_mm:.2f} mm | CPR frequency error: {n_cpr_error_per_minute:.2f} /min'
        print(msg)
        write_log_line(log_path, msg)
        
        if DEBUG:
            plt.imshow(rgb_imgs[p_indices[0]])
            plt.scatter(int(wrist_x[p_indices[0]]), int(wrist_y[p_indices[0]]), color='red')
            plt.title(f'Depth Image at Peak {p_indices[0]}')
            plt.show()  # Remove block=False for better control
            plt.clf()   # Clear the figure after showing it
    