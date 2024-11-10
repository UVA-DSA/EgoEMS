import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
parent_directory = os.path.abspath('.')
print(parent_directory)
sys.path.append(parent_directory)
from Tools.depth_sensor_processing import tools as depth_tools
import sys
import cv2
import extract_depth
import torch
import re

sample_rate = 30
window_duration = 5  # 5-second window
window_frames = window_duration * sample_rate  # Number of frames per 5-second window

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

GT_path = r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
data_path = r'D:\Final\exo_kinect_cpr_clips'
log_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\window_vision_log.txt'
debug_plots_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\window_debug_plots'
os.makedirs(debug_plots_path, exist_ok=True)

init_log(log_path)

for n in ['train_root', 'test_root', 'val_root']:
    data_dir = os.path.join(data_path, n, 'chest_compressions')
    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
    mkv_files = [file for file in os.listdir(data_dir) if file.endswith('.mkv')]

    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        json_data = json.load(open(json_path))
        mkv_file = [f for f in mkv_files if json_file.replace('_keypoints.json', '') in f][0]
        
        # rgb_imgs, depth_imgs = extract_depth.read_video(os.path.join(data_dir, mkv_file))

        # if not len(rgb_imgs) == len(json_data.keys()):
        #     continue

        keys = sorted([int(k) for k in json_data.keys()])
        wrist_x, wrist_y = [], []

        for k in keys:
            kpt_data = json_data.get(str(k), {})
            hands = kpt_data.get('hands', {})
            if len(hands) == 0:
                continue
            wrist_x.append(hands[0]['x'][0])
            wrist_y.append(hands[0]['y'][0])

        wrist_y = np.array(wrist_y, dtype=float)
        wrist_x = np.array(wrist_x, dtype=float)

        plt.title(f'Wrist Y keypoints for {json_file.split(".")[0]}')
        plt.plot(wrist_y)
        plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_wrist_y.png')
        plt.close()

        tensor_wrist_y = torch.tensor(wrist_y)
        try:
            filtered_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
        except Exception as e:
            print(f"Error in low pass filter: {e}")
            continue

        # Get start and end frames from filename for ground truth slicing
        match = re.search(r"(\d+\.\d+)_(\d+\.\d+)_exo\.mkv", mkv_file)
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

        for start in range(0, len(filtered_wrist_y), window_frames):
            end = start + window_frames
            if end > len(filtered_wrist_y):
                break  # Stop if the last window is incomplete

            # Predicted CPR cycles for the window
            window_wrist_y = filtered_wrist_y[start:end].numpy()
            p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(window_wrist_y, mul=1, show=False)
            n_cpr_window_pred = (len(p) + len(v)) * 0.5

            # Ground Truth CPR cycles for the window
            gt_window = gt_readings_for_clip[start:end]
            gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
            n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5




            # Log both predicted and GT CPR cycles for the window
            log_msg = (f"File: {json_file}, Window {start // window_frames + 1}, "
                       f"Predicted CPR cycles: {n_cpr_window_pred}, GT CPR cycles: {n_cpr_window_gt}")
            write_log_line(log_path, log_msg)
            print(log_msg)
