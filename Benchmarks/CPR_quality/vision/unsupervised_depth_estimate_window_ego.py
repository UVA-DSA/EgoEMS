import os
import numpy as np
import json
import sys
import cv2
import torch
import re
import shutil
from scipy.stats import zscore
import matplotlib.pyplot as plt

parent_directory = os.path.abspath('.')
print(parent_directory)
sys.path.append(parent_directory)
from Tools.depth_sensor_processing import tools as depth_tools

# Sample rate and window settings
sample_rate = 30
subsample_rate = 30
window_duration = 5  # 5-second window
window_frames = window_duration * sample_rate  # Number of frames per 5-second window

# Hyperparameter to scale x, y distances to mm
depth_scale_factor = 0.5  # Adjust this based on empirical calibration

def read_video(video_path):
    """Reads video and returns frames as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    cap.release()
    return frames


# Logging functions
def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)


def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')


# Define paths
GT_path = r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
data_path = r'D:\EgoExoEMS_CVPR2025\CPR Test\GoPro_CPR_Clips\ego_gopro_cpr_clips\test_root'
log_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\results\unsupervised_ego_depth_window_test_split_results.txt'
old_log_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\unsupervised_ego_depth_window_test_split_log.txt'
debug_plots_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\unsupervised_ego_depth_window_debug_plots'

# Setup directories
if os.path.exists(debug_plots_path):
    shutil.rmtree(debug_plots_path)
os.makedirs(debug_plots_path, exist_ok=True)

init_log(log_path)

# Processing files
data_dir = os.path.join(data_path, 'chest_compressions')
json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
mp4_files = [file for file in os.listdir(data_dir) if file.endswith('.mp4')]

for json_file in json_files:
    json_path = os.path.join(data_dir, json_file)
    json_data = json.load(open(json_path))
    mp4_file = [f for f in mp4_files if json_file.replace('_keypoints.json', '') in f][0]
    
    print("*" * 20)

    print(f"Processing file: {json_file}")
    print(f"Processing video: {mp4_file}")

    rgb_imgs = read_video(os.path.join(data_dir, mp4_file))
    if not len(rgb_imgs) == len(json_data.keys()):
        continue

    keys = sorted([int(k) for k in json_data.keys()])
    wrist_x, wrist_y = [], []

    matching_frame_indices = []
    for k in keys:
        kpt_data = json_data.get(str(k), {})
        hands = kpt_data.get('hands', {})
        if len(hands) == 0:
            continue
        wrist_x.append(hands[0]['x'][0])
        wrist_y.append(hands[0]['y'][0])
        matching_frame_indices.append(k)

    wrist_y = np.array(wrist_y, dtype=float)
    wrist_x = np.array(wrist_x, dtype=float)
    rgb_imgs = [rgb_imgs[i] for i in matching_frame_indices]

    tensor_wrist_y = torch.tensor(wrist_y)
    try:
        low_pass_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
    except Exception as e:
        print(f"Error in low pass filter: {e}")
        continue

    print(f"Number of wrist keypoints: {len(low_pass_wrist_y)}")

    # Get start and end frames from filename for ground truth slicing
    match = re.search(r"(\d+\.\d+)_(\d+\.\d+)_ego\.mp4", mp4_file)
    start_frame = end_frame = None
    if match:
        start_t = float(match.group(1))
        end_t = float(match.group(2))
        start_frame = int(start_t * sample_rate)
        end_frame = int(end_t * sample_rate)

    # Load ground truth data
    sbj = json_file.split('_')[0]
    t = json_file.split('_')[1][1:]
    gt_path = os.path.join(GT_path, sbj, 'cardiac_arrest', t, 'distance_sensor_data', 'sync_depth_sensor.csv')
    with open(gt_path, 'r') as file:
        gt_lines = file.readlines()

    gt_lines = np.array([float(line.strip()) for line in gt_lines[1:]])
    gt_readings_for_clip = gt_lines[start_frame:end_frame]

    for start in range(0, len(low_pass_wrist_y), window_frames):
        end = start + window_frames
        if end > len(low_pass_wrist_y):
            break

        # Process the window
        wrist_y_window = low_pass_wrist_y[start:end].numpy()
        wrist_x_window = wrist_x[start:end]

        print("Processing window: ", start, end, len(wrist_y_window))

        # Detect peaks and valleys
        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(wrist_y_window, mul=1, show=False)
        n_cpr_window_pred = (len(p) + len(v)) * 0.5

        # Make sure the number of peaks and valleys match
        min_length = min(len(p), len(v))
        p, v = p[:min_length], v[:min_length]  # Trim to equal length

        if min_length == 0:
            continue  # Skip window if no valid pairs

        # Extract peak and valley x, y coordinates
        wrist_x_p = wrist_x_window[p]
        wrist_y_p = wrist_y_window[p]
        wrist_x_v = wrist_x_window[v]
        wrist_y_v = wrist_y_window[v]

        # Compute Euclidean distances between peaks and valleys
        peak_valley_distances = np.sqrt((wrist_x_p - wrist_x_v) ** 2 + (wrist_y_p - wrist_y_v) ** 2)

        # Convert pixel distance to estimated depth using the scaling factor
        estimated_depths = peak_valley_distances * depth_scale_factor

        # Compute mean CPR depth for the window
        cpr_depth = np.mean(estimated_depths) if len(estimated_depths) > 0 else 0

        # Ground Truth CPR depth calculation
        gt_window = gt_readings_for_clip[start:end]
        gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
        n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

        peak_depths_gt = gt_window[gt_peaks]
        valley_depths_gt = gt_window[gt_valleys]
        l = min(len(peak_depths_gt), len(valley_depths_gt))
        gt_cpr_depth = float((peak_depths_gt[:l] - valley_depths_gt[:l]).mean()) if l > 0 else 0

        print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")
        print(f"Predicted CPR Depth (mm): {cpr_depth:.2f}")

        # Log results
        log_msg = f"File: {json_file}, Window {start // window_frames + 1}, Predicted CPR depth: {cpr_depth:.2f}mm, GT CPR depth: {gt_cpr_depth:.2f}mm"
        write_log_line(log_path, log_msg)

    rgb_imgs.clear()
    del rgb_imgs

    print("*" * 20)
