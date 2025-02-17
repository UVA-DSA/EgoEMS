import os
import numpy as np
import json
import sys
import cv2
import torch
import re
import shutil
from scipy.stats import zscore
from torchvision import transforms
from PIL import Image

import time

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


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    midas_transform = midas_transforms.dpt_transform
else:
    midas_transform = midas_transforms.small_transform


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

def estimate_depth_from_rgb(rgb_img):
    """Estimates depth from an RGB image using MiDaS."""
    # img = Image.fromarray(rgb_img)
    # img = Image.fromarray(rgb_img)
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # make image smaller 384, 384 for faster inference
    img = cv2.resize(img, (384, 384))
    input_batch = midas_transform(img).to(device)

    with torch.no_grad():
        prediction  = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # plt.figure(figsize=(6, 6))
    # plt.imshow(depth_map, cmap='inferno')  # Use a perceptually uniform colormap
    # plt.colorbar(label="Depth (mm)")
    # plt.title("Estimated Depth Map")
    # plt.show()  # This is required to display the image


    return depth_map

# Logging functions
def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')

# Define base paths using environment variables or user input

# Windows environment
# GT_path = r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
# data_path = r'D:\EgoExoEMS_CVPR2025\CPR Test\GoPro_CPR_Clips\ego_gopro_cpr_clips\test_root'
# log_path = rf"E:\EgoExoEMS\Benchmarks\CPR_quality\vision\results\supervised_{model_type}_ego_depth_window_test_split_results.txt"
# old_log_path = rf"E:\EgoExoEMS\Benchmarks\CPR_quality\vision\supervised_{model_type}_ego_depth_window_test_split_log.txt"
# debug_plots_path = rf"E:\EgoExoEMS\Benchmarks\CPR_quality\vision\supervised_{model_type}_ego_depth_window_debug_plots"


# Linux environment
BASE_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025"  # Set an environment variable or update this
BASE_REPO_DIR = "/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS"
GT_path = os.path.join(BASE_DIR, "Dataset", "Final")
data_path = os.path.join(BASE_DIR, "Dataset", "GoPro_CPR_Clips", "ego_gopro_cpr_clips", "test_root")
log_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", f"supervised_{model_type}_ego_depth_window_test_split_results.txt")
debug_plots_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", f"supervised_{model_type}_ego_depth_window_debug_plots")


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

    print("*"*20)

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

        start_t = time.time()
        # Process the entire window
        wrist_y_window = low_pass_wrist_y[start:end].numpy()
        wrist_x_window = wrist_x[start:end]

        print("Processing window: ", start, end, len(wrist_y_window))



        # Detect peaks and valleys
        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(wrist_y_window, mul=1, show=False)
        n_cpr_window_pred = (len(p) + len(v)) * 0.5

        # Extract peak and valley x, y coordinates
        wrist_x_p = wrist_x_window[p]
        wrist_y_p = wrist_y_window[p]
        wrist_x_v = wrist_x_window[v]
        wrist_y_v = wrist_y_window[v]

        # Generate depth maps for the window
        depth_maps = [estimate_depth_from_rgb(img) for img in rgb_imgs]

        print("X peaks before scaling: ", wrist_x_p)

        # scale wrist_x_p to depth map size
        wrist_x_p = [int(x * depth_maps[i].shape[1] / rgb_imgs[i].shape[1]) for i, x in enumerate(wrist_x_p)]
        wrist_x_v = [int(x * depth_maps[i].shape[1] / rgb_imgs[i].shape[1]) for i, x in enumerate(wrist_x_v)]
        wrist_y_p = [int(y * depth_maps[i].shape[0] / rgb_imgs[i].shape[0]) for i, y in enumerate(wrist_y_p)]
        wrist_y_v = [int(y * depth_maps[i].shape[0] / rgb_imgs[i].shape[0]) for i, y in enumerate(wrist_y_v)]

        print("X peaks after scaling: ", wrist_x_p)

        # Extract depth values for peaks and valleys
        peak_depths = np.array([
            depth_maps[i][int(y), int(x)] for i, (x, y) in enumerate(zip(wrist_x_p, wrist_y_p))
            if 0 <= x < depth_maps[i].shape[1] and 0 <= y < depth_maps[i].shape[0]
        ])
        valley_depths = np.array([
            depth_maps[i][int(y), int(x)] for i, (x, y) in enumerate(zip(wrist_x_v, wrist_y_v))
            if 0 <= x < depth_maps[i].shape[1] and 0 <= y < depth_maps[i].shape[0]
        ])

        # Compute CPR depth for the window
        if len(peak_depths) > 0 and len(valley_depths) > 0:
            min_length = min(len(peak_depths), len(valley_depths))
            cpr_depth = np.mean(peak_depths[:min_length] - valley_depths[:min_length])
        else:
            cpr_depth = 0


        # Ground Truth CPR cycles for the window
        gt_window = gt_readings_for_clip[start:end]
        gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
        n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

        peak_depths_gt=gt_window[gt_peaks]
        valley_depths_gt=gt_window[gt_valleys]
        l=min(len(peak_depths_gt),len(valley_depths_gt))
        gt_cpr_depth=float((peak_depths_gt[:l]-valley_depths_gt[:l]).mean())

        print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")
        print(f"Predicted CPR Depth (mm): {cpr_depth:.2f}")

        end_t = time.time()
        inference_time = end_t - start_t
        print(f"Time taken for window: {inference_time} seconds")
        # Log results
        log_msg = f"File: {json_file}, Window {start // window_frames + 1}, Predicted CPR depth: {cpr_depth:.2f}mm, GT CPR depth: {gt_cpr_depth:.2f}mm, Inference time: {inference_time} seconds"
        write_log_line(log_path, log_msg)

    print("*"*20)   

    rgb_imgs.clear()
    del rgb_imgs
