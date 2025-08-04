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
import open3d as o3d
import shutil

import time

from scipy.stats import zscore

sample_rate = 30
subsample_rate = 30
window_duration = 5  # 5-second window
window_frames = window_duration * sample_rate  # Number of frames per 5-second window

CANON_K = np.array([[615.873673811006, 0, 640.803032851225],
                    [0, 615.918359977960, 365.547839233105],
                    [0, 0, 1]])

width = 1280
height = 720
# Define Open3D intrinsic parameters based on CANON_K
# intrinsics = o3d.camera.PinholeCameraIntrinsic(
#     width=1280, height=720,  # replace with actual Kinect frame resolution
#     fx=CANON_K[0, 0], fy=CANON_K[1, 1],
#     cx=CANON_K[0, 2], cy=CANON_K[1, 2]
# )

intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=CANON_K[0, 0], fy=CANON_K[1, 1], cx=CANON_K[0, 2], cy=CANON_K[1, 2])



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



def get_XYZ(x, y, depth, intrinsics):
    """Transform 2D (x, y) pixel coordinates to 3D space using the depth value and camera intrinsics."""
    X = (x - intrinsics.intrinsic_matrix[0, 2]) * depth / intrinsics.intrinsic_matrix[0, 0]
    Y = (y - intrinsics.intrinsic_matrix[1, 2]) * depth / intrinsics.intrinsic_matrix[1, 1]
    Z = depth
    return np.array([X, Y, Z])

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

GT_path = r'F:\EgoEMS Dataset\Dataset\Final'
data_path = r'G:\Research\EgoEMS\Dataset\CPR\exo_kinect_cpr_clips'
log_path = r'F:\repos\EgoExoEMS\Benchmarks\CPR_quality\vision\logs\july14_final_exocentric_kinect_cpr_rate_window_results.txt'
debug_plots_path = r'F:\repos\EgoExoEMS\Benchmarks\CPR_quality\vision\debug_plots\july14_final_exocentric_kinect_cpr_rate_window_debug_plots'


# delete the directory if it already exists
if os.path.exists(debug_plots_path):
    shutil.rmtree(debug_plots_path)

os.makedirs(debug_plots_path, exist_ok=True)


init_log(log_path)


# for n in ['train_root', 'test_root', 'val_root']:
for n in ['test_root']:
    data_dir = os.path.join(data_path, n, 'chest_compressions')
    json_files = [file for file in os.listdir(data_dir) if file.endswith('_resized_640x480_keypoints.json')]
    mkv_files = [file for file in os.listdir(data_dir) if file.endswith('.mkv')]

    for json_file in json_files:

               # check if the file has already been processed
        # if any(json_file in line for line in old_log):
        #     print(f"Skipping {json_file} as it has already been processed.")
        #     continue

        # if(json_file not in ['ms1_t5_ks4_58_keypoints.json']):
        #     continue


        json_path = os.path.join(data_dir, json_file)
        json_data = json.load(open(json_path))
        mkv_file = [f for f in mkv_files if json_file.replace('_resized_640x480_keypoints.json', '') in f][0]

 
        # if "P14" not in mkv_file:
        #     continue

        print("*"*20)
        print(f"Processing file: {json_file}")
        
        rgb_imgs, depth_imgs = extract_depth.read_video(os.path.join(data_dir, mkv_file))

        print("Number of RGB images: ", len(rgb_imgs))
        print("Number of depth images: ", len(depth_imgs))
        print("Number of keypoints: ", len(json_data.keys()))

        # check if depth imgs are empty
        if len(depth_imgs) == 0:
            print(f"No depth images found for {mkv_file}. Skipping...")
            continue

        if not len(rgb_imgs) == len(json_data.keys()) and not len(depth_imgs) == len(json_data.keys()):
            print(f"Number of frames in RGB and depth videos do not match the number of keypoints in {json_file}")
            
            # get only keypoints that have corresponding frames in the RGB and depth videos
            json_data = {k: v for k, v in json_data.items() if int(k) < len(rgb_imgs) and int(k) < len(depth_imgs)}
            print("Number of keypoints after filtering: ", len(json_data.keys()))



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

        # scale wrist coordinates to match the RGB and depth image dimensions
        wrist_y = (wrist_y * depth_imgs[0].shape[0] / 480).astype(int)
        wrist_x = (wrist_x * depth_imgs[0].shape[1] / 640).astype(int)

        # Use the matching_frame_indices to retrieve the corresponding RGB and depth frames
        matching_rgb_frames = [rgb_imgs[i] for i in matching_frame_indices]
        matching_depth_frames = [depth_imgs[i] for i in matching_frame_indices]

        rgb_imgs = matching_rgb_frames
        depth_imgs = matching_depth_frames

        print("Number of RGB images: ", len(rgb_imgs))
        print("Number of depth images: ", len(depth_imgs))
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

        print("number of wrist keypoints: ", len(low_pass_wrist_y))
        for start in range(0, len(low_pass_wrist_y), window_frames):
            end = start + window_frames

            print("*"*20)
            start_t = time.time()

            window_num = start // window_frames + 1
            print(f"Processing window {window_num}")
            print(f"Start: {start}, End: {end}")

            if end > len(low_pass_wrist_y):
                break  # Stop if the last window is incomplete

            start_t = time.time()

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
            depth_imgs_window = depth_imgs[start:end]
            depth_imgs_window = np.array(depth_imgs_window)
            filtered_depth_imgs_window = depth_imgs_window[final_filtered_indices]



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

            gt_cpr_rate_per_min = n_cpr_window_gt * 60 / window_duration
            pred_cpr_rate_per_min = n_cpr_window_pred * 60 / window_duration

            end_t = time.time()
            inference_time = end_t - start_t
            print(f"Inference time for the window: {inference_time} seconds")

            log_msg = (f"File:{json_file},Window:{window_num},Predicted_CPR_Rate:{pred_cpr_rate_per_min:.2f},GT_CPR_Rate:{gt_cpr_rate_per_min:.2f},InferenceTimeSeconds:{inference_time}")

            # log_msg = (f"File: {json_file}, Window {window_num}, "
            #            f"Predicted CPR cycles: {n_cpr_window_pred}, GT CPR cycles: {n_cpr_window_gt}")
            write_log_line(log_path, log_msg)
            print(log_msg)


            # print("number of filtered wrist keypoints: ", len(filtered_wrist_x))
            # print("number of filtered depth images in the window: ", len(filtered_depth_imgs_window))

            # # save the depth images for the window for visualization
            # # for i, depth_img in enumerate(window_depth_imgs):
            # #     save_path = os.path.join(debug_plots_path, f"{json_file.split('.')[0]}_window_{window_num}_depth_frame_{i}.png")
            # #     cv2.imwrite(save_path, depth_img)

            # # Get the depth values for the wrist keypoints

            # # x wrist keypoints for the window


            # # Filter indices based on outlier detection for both X and Y

            # p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(filtered_wrist_y, mul=1, show=False)

            # filtered_depth_imgs_p=filtered_depth_imgs_window[p]
            # filtredepth_imgs_v=filtered_depth_imgs_window[v]


            # wrist_x_p=filtered_wrist_x[p]
            # wrist_y_p=filtered_wrist_y[p]
            # wrist_x_v=filtered_wrist_x[v]
            # wrist_y_v=filtered_wrist_y[v]


            # peak_z_vals, valley_z_vals = [], []
            # peak_3d_points, valley_3d_points = [], []

            #     # Convert 2D wrist keypoints to 3D for peaks
            # for i, depth_img in enumerate(filtered_depth_imgs_p):
            #     depth = depth_img[int(wrist_y_p[i]), int(wrist_x_p[i])]  # Get depth at wrist keypoint

            #     # draw a circle around the wrist keypoint
            #     depth_img_copy = depth_img.copy()
            #     depth_img_copy = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_copy, alpha=0.03), cv2.COLORMAP_JET)

            #     cv2.circle(depth_img_copy, (int(wrist_x_p[i]), int(wrist_y_p[i])), radius=5, color=(0, 255, 0), thickness=-1)
            #     save_path = os.path.join(debug_plots_path, f"{json_file.split('.')[0]}_window_{window_num}_peak_{i}.png")
            #     cv2.imwrite(save_path, depth_img_copy)

            #     # print(f"Peak {i} depth: {depth}mm")
            #     xyz = get_XYZ(wrist_x_p[i], wrist_y_p[i], depth, intrinsics)
            #     # print(f"Peak {i} 3D point: {xyz}")

            #     peak_3d_points.append(xyz)

            # # Convert 2D wrist keypoints to 3D for valleys
            # for i, depth_img in enumerate(filtredepth_imgs_v):
            #     depth = depth_img[int(wrist_y_v[i]), int(wrist_x_v[i])]  # Get depth at wrist keypoint

            #     # draw a circle around the wrist keypoint
            #     depth_img_copy = depth_img.copy()
            #     # colorize the depth image
            #     depth_img_copy = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_copy, alpha=0.03), cv2.COLORMAP_JET)
            #     cv2.circle(depth_img_copy, (int(wrist_x_v[i]), int(wrist_y_v[i])), radius=5, color=(0, 255, 0), thickness=-1)
            #     save_path = os.path.join(debug_plots_path, f"{json_file.split('.')[0]}_window_{window_num}_valley_{i}.png")
            #     cv2.imwrite(save_path, depth_img_copy)

            #     # print(f"Valley {i} depth: {depth}mm")
            #     # print(f"Valley {i} wrist x: {wrist_x_v[i]}, wrist y: {wrist_y_v[i]}")
            #     xyz = get_XYZ(wrist_x_v[i], wrist_y_v[i], depth, intrinsics)
            #     # print(f"Valley {i} 3D point: {xyz}")

            #     valley_3d_points.append(xyz)


            # # remove any outliers in the peaks and valleys
            # print("Number of peaks: ", len(peak_3d_points))
            # print("Number of valleys: ", len(valley_3d_points))

            # # print("Peaks: ", peak_3d_points)
            # # print("Valleys: ", valley_3d_points)

            #             # Apply the outlier removal function to peak and valley points
            # # filtered_peak_3d_points = remove_outliers_3d_pairs(peak_3d_points)
            # # filtered_valley_3d_points = remove_outliers_3d_pairs(valley_3d_points)
            # # print("3d peaks: ", peak_3d_points)
            # # print("3d valleys: ", valley_3d_points)

            # # interpolate zero valley and peak points
            # peak_3d_points = interpolate_zero_coords(peak_3d_points)
            # valley_3d_points = interpolate_zero_coords(valley_3d_points)
            # print("-"*20)

            # # print("3d peaks after interpolation: ", peak_3d_points)
            # # print("3d valleys after interpolation: ", valley_3d_points)


            # # Calculate Euclidean distances between consecutive peaks and valleys
            # distances = []
            # for peak, valley in zip(peak_3d_points, valley_3d_points):
            #     # skip None values
            #     if peak is None or valley is None:
            #         continue
            #     # if(np.array_equal(peak, [0, 0, 0]) or np.array_equal(valley, [0, 0, 0])):
            #     #     continue
                
            #     dist = np.linalg.norm(peak - valley)
            #     # print("distance: ", dist)
            #     distances.append(dist)


            # # mean distance between peaks and valleys
            # print("Distances: ", distances)


            # # remove outliers from the distances
            # distances = np.array(distances)

            # # remove outliers from the distances
            # distances = remove_outliers_zscore(distances, threshold=1.5)

            # cpr_depth = np.mean(distances)

            # print(f"Predicted Mean distance between peaks and valleys: {cpr_depth}mm")
            # print(f"Ground truth CPR depth: {gt_cpr_depth}mm")

            # end_t = time.time()
            # inference_time = end_t - start_t
            # print(f"Inference time for the window: {inference_time} seconds")

            # log_msg = (f"File:{json_file},Window:{window_num},Predicted_CPR_Depth:{cpr_depth:.2f},GT_CPR_Depth:{gt_cpr_depth:.2f},InferenceTimeSeconds:{inference_time}")

            # write_log_line(log_path, log_msg)
            # print("*"*20)


        # Clear large arrays after processing
        rgb_imgs.clear()
        depth_imgs.clear()

        # Explicitly delete variables with large data
        del rgb_imgs, depth_imgs

            # Log distances for debugging
            # log_msg = (f"File: {json_file}, Window {start // window_frames + 1}, "
            #         f"Distances: {distances}")
            # write_log_line(log_path, log_msg)
            # print(log_msg)



            # get the depth values for the peaks and valleys

            # peakXYZ_p=[]
            # for idx in range(len(depth_imgs_p)):
            #     depth_img=depth_imgs_p[idx]
            #     d=int(depth_img[int(wrist_y_p[idx]),int(wrist_x_p[idx])])
            #     X,Y,Z=get_XYZ(int(wrist_x_p[idx]),int(wrist_y_p[idx]),d,CANON_K)
            #     print(f"Peaks in Kinect: X: {X}, Y: {Y}, Z: {Z}")
            #     peakXYZ_p.append([float(X),float(Y),float(Z)])
            # peakXYZ_v=[]
            # for idx in range(len(depth_imgs_v)):
            #     depth_img=depth_imgs_v[idx]
            #     d=int(depth_img[int(wrist_y_v[idx]),int(wrist_x_v[idx])])
            #     X,Y,Z=get_XYZ(int(wrist_x_v[idx]),int(wrist_y_v[idx]),d,CANON_K)
            #     print(f"Valleys in Kinect: X: {X}, Y: {Y}, Z: {Z}")
            #     peakXYZ_v.append([float(X),float(Y),float(Z)])

            # l=min(len(peakXYZ_p),len(peakXYZ_v))

            # dist_list=[]
            # for idx in range(l):
            #     p_=np.array(peakXYZ_p[idx])
            #     v_=np.array(peakXYZ_v[idx])
            #     dist=np.sum((p_-v_)**2)**0.5
            #     dist_list.append(float(dist))
            # cpr_depth=float(np.mean(dist_list))



            # depth_val_window  = []
            # # get the depth values for the window using the wrist keypoints for the window
            # for i, depth_img in enumerate(window_depth_imgs):
            #     x, y = int(wrist_x_window[i]), int(wrist_y_window[i])
            #     depth_val = depth_img[y, x]

            #     X, Y, Z = get_XYZ(x, y, depth_val, CANON_K)
            #     print(f"X: {X}, Y: {Y}, Z: {Z}")
            #     depth_val_window.append(depth_val)
            #     print(f"Depth value for frame {i} in window {window_num}: {depth_val}mm")

            # # plot the depth values for the window
            # plt.plot(depth_val_window)
            # plt.title(f'Kinect Depth values for {json_file.split(".")[0]} window {window_num}')
            # plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_window_{window_num}_depth_values.png')
            # plt.close()


            # Subsample and save visualized frames with wrist keypoints
            # Subsample and save visualized frames with wrist keypoints
            # for i,depth_img in enumerate(window_depth_imgs):

            #     # Use the actual depth image for each frame index
            #     x, y = int(wrist_x_window[i]), int(wrist_y_window[i])

            #     depth_img_copy = depth_img.copy()


            #     # Ensure coordinates are within image bounds
            #     if 0 <= y < depth_img.shape[0] and 0 <= x < depth_img.shape[1]:
            #         # Create an overlayed color image for visualization
            #         depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_copy, alpha=0.03), cv2.COLORMAP_JET)
            #         cv2.circle(depth_colored, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

            #         # Save the frame with the overlayed wrist keypoint
            #         save_path = os.path.join(debug_plots_path, f"{json_file.split('.')[0]}_window_{window_num}_frame_{i}.png")
            #         cv2.imwrite(save_path, depth_colored)

    #     break
    # break