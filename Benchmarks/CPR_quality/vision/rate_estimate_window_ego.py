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
import torch
import re
import shutil

from scipy.stats import zscore

sample_rate = 30
subsample_rate = 30
window_duration = 5  # 5-second window
window_frames = window_duration * sample_rate  # Number of frames per 5-second window


# function to read video and load frames as numpy arrays using opencv
def read_video(video_path):
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames

    # Read frames from the video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no more frames

        # Convert the frame to RGB format (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

    # Release the video capture object
    cap.release()

    return frames





def interpolate_zero_coords(coords):
    """
    Replace any (0, 0, 0) points in the coords list with the most recent non-zero point.
    If the list starts with zero points, they are forward-filled with the first valid (non-zero) point.

    Args:
    coords (list of np.array): List of 3D valley points (each point is an array [X, Y, Z]).

    Returns:
    list of np.array: List of 3D valley points where any (0, 0, 0) entries are replaced by the nearest valid point.
    """
    # Keep track of the last valid (non-zero) valley point
    last_valid_valley = None

    for i in range(len(coords)):
        # Check if the current valley is a zero point
        if np.array_equal(coords[i], [0, 0, 0]):
            # Replace with the last valid valley point if available
            if last_valid_valley is not None:
                coords[i] = last_valid_valley
        else:
            # Update the last valid valley if the current point is non-zero
            last_valid_valley = coords[i]

    # Forward-fill initial zero points with the last valid point if needed
    for i in range(len(coords)):
        if np.array_equal(coords[i], [0, 0, 0]):
            coords[i] = last_valid_valley
        else:
            break  # Stop forward-filling once a valid point is found

    return coords



def remove_outliers_zscore(data, threshold=2):
    """
    Remove outliers from a list based on the Z-score method.

    Args:
    data (list of float): List of values from which to remove outliers.
    threshold (float): Z-score threshold for defining outliers. Default is 2.

    Returns:
    list of float: Filtered list without outliers.
    """
    # Calculate the z-scores for each data point
    z_scores = zscore(data)

    # Filter out data points where the absolute z-score is below the threshold
    filtered_data = [val for val, z in zip(data, z_scores) if abs(z) < threshold]

    return filtered_data

# Function to remove outliers
def remove_outliers(data, threshold=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_indices = np.abs(data - mean) <= threshold * std_dev
    return filtered_indices

def remove_outliers_3d_pairs(peak_points, valley_points, z_threshold=2):
    """
    Remove outlier pairs from peak and valley 3D points based on the average
    Euclidean distance from the mean of both peak and valley points together.

    Args:
    peak_points (list of np.array): List of 3D peak points (each point is [X, Y, Z]).
    valley_points (list of np.array): List of 3D valley points (each point is [X, Y, Z]).
    z_threshold (float): Threshold for the z-score to filter outlier pairs.

    Returns:
    filtered_peak_points (list of np.array): List of filtered 3D peak points.
    filtered_valley_points (list of np.array): List of filtered 3D valley points.
    """
    if len(peak_points) == 0 or len(valley_points) == 0 or len(peak_points) != len(valley_points):
        return peak_points, valley_points

    # Remove pairs with zero depth values
    valid_peak_points = []
    valid_valley_points = []
    for peak, valley in zip(peak_points, valley_points):
        if peak[2] != 0 and valley[2] != 0:  # Ensure both peak and valley have non-zero Z values
            valid_peak_points.append(peak)
            valid_valley_points.append(valley)

    # Calculate pairwise distances on the filtered non-zero points
    distances = [np.linalg.norm(peak - valley) for peak, valley in zip(valid_peak_points, valid_valley_points)]
    mean_dist = np.mean(distances)
    std_dev = np.std(distances)

    # Filter out outlier pairs based on distance z-score
    filtered_peak_points = []
    filtered_valley_points = []
    for peak, valley, distance in zip(valid_peak_points, valid_valley_points, distances):
        z_score = (distance - mean_dist) / std_dev if std_dev > 0 else 0
        if abs(z_score) <= z_threshold:
            filtered_peak_points.append(peak)
            filtered_valley_points.append(valley)

    return filtered_peak_points, filtered_valley_points




def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')

GT_path = r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
data_path = r'D:\EgoExoEMS_CVPR2025\CPR Test\GoPro_CPR_Clips'
log_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\outlier_depth_window_ego_vision_log_pv_interpolated_final.txt'
old_log_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\ego_window_vision_log_backup.txt'
debug_plots_path = r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\ego_depth_window_debug_plots'

# delete the directory if it already exists
if os.path.exists(debug_plots_path):
    shutil.rmtree(debug_plots_path)

os.makedirs(debug_plots_path, exist_ok=True)


init_log(log_path)

# 
# Handle old log file safely
if os.path.exists(old_log_path):
    with open(old_log_path, 'r') as file:
        old_log = file.readlines()
else:
    old_log = []


data_dir = os.path.join(data_path, 'chest_compressions')
json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
mp4_files = [file for file in os.listdir(data_dir) if file.endswith('.mp4')]

for json_file in json_files:

    json_path = os.path.join(data_dir, json_file)
    json_data = json.load(open(json_path))
    mp4_file = [f for f in mp4_files if json_file.replace('_keypoints.json', '') in f][0]

    print("Processing file: ", json_file)
    print("Processing video: ", mp4_file)

    # read the video file and get the frames
    rgb_imgs = read_video(os.path.join(data_dir, mp4_file))



    if not len(rgb_imgs) == len(json_data.keys()):
        continue

    keys = sorted([int(k) for k in json_data.keys()])
    wrist_x, wrist_y = [], []


    # Store the frame indices where wrist keypoints were detected
    matching_frame_indices = []

    for k in keys:
        kpt_data = json_data.get(str(k), {})
        hands = kpt_data.get('hands', {})
        if len(hands) == 0:
            continue
        # Append wrist coordinates and store the frame index
        wrist_x.append(hands[0]['x'][0])
        wrist_y.append(hands[0]['y'][0])
        matching_frame_indices.append(k)  # Store frame index for detected wrist

    # Convert wrist coordinates to NumPy arrays for easier manipulation
    wrist_y = np.array(wrist_y, dtype=float)
    wrist_x = np.array(wrist_x, dtype=float)

    # Use the matching_frame_indices to retrieve the corresponding RGB and depth frames
    matching_rgb_frames = [rgb_imgs[i] for i in matching_frame_indices]

    rgb_imgs = matching_rgb_frames

    print("Number of RGB images: ", len(rgb_imgs))
    print("Number of wrist keypoints: ", len(wrist_y))


    # plt.title(f'Wrist Y keypoints for {json_file.split(".")[0]}')
    # plt.plot(wrist_y)
    # plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_wrist_y.png')
    # plt.show(block=True)
    # plt.close()

    # plt.title(f'Wrist X keypoints for {json_file.split(".")[0]}')
    # plt.plot(wrist_x)
    # plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_wrist_x.png')
    # plt.show(block=True)
    # plt.close()


    # plot the wrist keypoints on the depth images


    tensor_wrist_y = torch.tensor(wrist_y)
    try:
        low_pass_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
    except Exception as e:
        print(f"Error in low pass filter: {e}")
        continue

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

    print("number of wrist keypoints: ", len(low_pass_wrist_y))
    for start in range(0, len(low_pass_wrist_y), window_frames):
        end = start + window_frames
        if end > len(low_pass_wrist_y):
            break  # Stop if the last window is incomplete

        # Predicted CPR cycles for the window
        wrist_y_window = low_pass_wrist_y[start:end].numpy()
        wrist_x_window = wrist_x[start:end]

                    # Filter indices based on outlier detection for both X and Y
        x_filtered_indices = remove_outliers(wrist_x_window, threshold=1)
        y_filtered_indices = remove_outliers(wrist_y_window, threshold=1)

        # Combine filters to maintain synchronization
        final_filtered_indices = x_filtered_indices & y_filtered_indices
        # print("Final filtered indices: ", final_filtered_indices, len(final_filtered_indices))

        # Filter the arrays
        filtered_wrist_x = wrist_x_window[final_filtered_indices]
        filtered_wrist_y = wrist_y_window[final_filtered_indices]

        # Output the filtered data
        # print("Filtered Wrist X:", filtered_wrist_x)
        # print("Filtered Wrist Y:", filtered_wrist_y)

        # filter the depth images for the window
        rgb_imgs_window = rgb_imgs[start:end]
        rgb_imgs_window = np.array(rgb_imgs_window)
        filtered_rgb_imgs_window = rgb_imgs_window[final_filtered_indices]



        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(wrist_y_window, mul=1, show=False)
        n_cpr_window_pred = (len(p) + len(v)) * 0.5

        # Ground Truth CPR cycles for the window
        gt_window = gt_readings_for_clip[start:end]
        gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
        n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

        peak_depths_gt=gt_window[gt_peaks]
        valley_depths_gt=gt_window[gt_valleys]
        l=min(len(peak_depths_gt),len(valley_depths_gt))
        gt_cpr_depth=float((peak_depths_gt[:l]-valley_depths_gt[:l]).mean())

        # Log both predicted and GT CPR cycles for the window
        window_num = start // window_frames + 1

        log_msg = (f"File: {json_file}, Window {window_num}, "
                    f"Predicted CPR cycles: {n_cpr_window_pred}, GT CPR cycles: {n_cpr_window_gt}")
        write_log_line(log_path, log_msg)
        print(log_msg)


        print("number of filtered wrist keypoints: ", len(filtered_wrist_x))
        print("number of filtered rgb images in the window: ", len(filtered_rgb_imgs_window))


        # Filter indices based on outlier detection for both X and Y

        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(filtered_wrist_y, mul=1, show=False)

        filtered_rgb_imgs_p=filtered_rgb_imgs_window[p]
        filtredepth_imgs_v=filtered_rgb_imgs_window[v]


        wrist_x_p=filtered_wrist_x[p]
        wrist_y_p=filtered_wrist_y[p]
        wrist_x_v=filtered_wrist_x[v]
        wrist_y_v=filtered_wrist_y[v]


        distances = []
        cpr_depth = np.mean(distances)

        print(f"Predicted Mean distance between peaks and valleys: {cpr_depth}mm")
        print(f"Ground truth CPR depth: {gt_cpr_depth}mm")

        log_msg = (f"File: {json_file}, Window {window_num}, "
                    f"Predicted CPR depth: {cpr_depth}mm, GT CPR depth: {gt_cpr_depth}mm")
        write_log_line(log_path, log_msg)

    # Clear large arrays after processing
    rgb_imgs.clear()

    # Explicitly delete variables with large data
    del rgb_imgs

       