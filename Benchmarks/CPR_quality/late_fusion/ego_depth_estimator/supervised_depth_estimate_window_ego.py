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
from scipy.stats import zscore

import open3d as o3d
import time

import matplotlib.pyplot as plt



parent_directory = os.path.abspath('.')
print(parent_directory)
sys.path.append(parent_directory)

from tools import tools as depth_tools

# Sample rate and window settings
sample_rate = 30
subsample_rate = 30
window_duration = 5  # 5-second window
window_frames = int(window_duration * sample_rate)  # Number of frames per 5-second window

DEBUG = True

# Intrinsic parameters for the GoPro camera
GOPRO_K_2704p = np.array([[1229.44, 0, 1333.15],
                          [0, 925.32, 753.23],
                          [0, 0, 1]])

width = 2704
height = 1520
intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height,
                                               fx=GOPRO_K_2704p[0, 0],
                                               fy=GOPRO_K_2704p[1, 1],
                                               cx=GOPRO_K_2704p[0, 2],
                                               cy=GOPRO_K_2704p[1, 2])


def get_XYZ(x, y, depth, intrinsics):
    """Transform 2D (x, y) pixel coordinates to 3D space using the depth value and camera intrinsics."""
    X = (x - intrinsics.intrinsic_matrix[0, 2]) * depth / intrinsics.intrinsic_matrix[0, 0]
    Y = (y - intrinsics.intrinsic_matrix[1, 2]) * depth / intrinsics.intrinsic_matrix[1, 1]
    Z = depth
    return np.array([X, Y, Z])




def initialize_midas_model(model_type, device):

    # Load MiDaS depth estimation model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    print(f"[INFO] Egocentric depth estimation using MiDaS model: {model_type}")

    if model_type == "DPT_Large":
        SCALE_FACTOR = 23.33
    else:
        SCALE_FACTOR = 0.676

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        midas_transform = midas_transforms.dpt_transform
    else:
        midas_transform = midas_transforms.small_transform
        
    print(f"[INFO] Using MiDaS transform: {midas_transform}")
    return midas, midas_transform, SCALE_FACTOR


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


def estimate_depth_from_rgb(rgb_img, midas, midas_transform, device, output_path=None):
    """
    Estimates depth from an RGB image using MiDaS.
    If output_path is provided and DEBUG=True, a matplotlib plot (depth-only) will be saved there
    with a colorbar and axis labels.
    """
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # If requested, save a debug figure of the depth map alone (with colorbar, etc.)
    if DEBUG and output_path is not None:
        # Normalize depth to [0,1] and invert it
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val - min_val < 1e-6:
            max_val = min_val + 1e-6
        normalized_depth = (depth_map - min_val) / (max_val - min_val)
        inverted_depth = 1 - normalized_depth  # Invert depth to match colormap expectation

        img_h, img_w = depth_map.shape
        aspect_ratio = img_w / img_h

        fig, ax = plt.subplots(figsize=(6 * aspect_ratio, 6))
        im = ax.imshow(inverted_depth, cmap='inferno')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Relative Depth")
        ax.set_title("Estimated Depth Map")
        ax.set_aspect("auto")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    return depth_map


def save_side_by_side_figure(rgb_img, depth_map, output_path, wrist_xy=None):
    """
    Save a single figure with:
      - Left: RGB image (with a green circle marking wrist_xy if provided).
      - Right: Depth map (normalized & inverted) + colorbar.
    """
    rgb_display = rgb_img.copy()
    if wrist_xy is not None:
        wx, wy = wrist_xy
        if 0 <= wx < rgb_display.shape[1] and 0 <= wy < rgb_display.shape[0]:
            cv2.circle(rgb_display, (wx, wy), 8, (0, 255, 0), -1)

    min_val, max_val = depth_map.min(), depth_map.max()
    if max_val - min_val < 1e-6:
        max_val = min_val + 1e-6
    normalized = (depth_map - min_val) / (max_val - min_val)
    inverted = 1.0 - normalized

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(rgb_display)
    axs[0].set_title("RGB Image")
    axs[0].set_xlabel("Pixel X")
    axs[0].set_ylabel("Pixel Y")
    axs[0].axis("on")

    im = axs[1].imshow(inverted, cmap='inferno')
    axs[1].set_title("Depth Map")
    axs[1].axis("on")
    axs[1].set_xlabel("Pixel X")
    axs[1].set_ylabel("Pixel Y")

    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Relative Depth")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_top_bottom_figure(rgb_img, depth_map, output_path, wrist_xy=None):
    """
    Save a single figure with:
      - Top: RGB image (with a green circle marking wrist_xy if provided).
      - Bottom: Depth map (normalized & inverted) + colorbar.
    """
    rgb_display = rgb_img.copy()
    if wrist_xy is not None:
        wx, wy = wrist_xy
        if 0 <= wx < rgb_display.shape[1] and 0 <= wy < rgb_display.shape[0]:
            cv2.circle(rgb_display, (wx, wy), 8, (0, 255, 0), -1)

    min_val, max_val = depth_map.min(), depth_map.max()
    if max_val - min_val < 1e-6:
        max_val = min_val + 1e-6

    normalized = (depth_map - min_val) / (max_val - min_val)
    inverted = 1.0 - normalized

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].imshow(rgb_display)
    axs[0].set_title("RGB Image")
    axs[0].set_xlabel("Pixel X")
    axs[0].set_ylabel("Pixel Y")
    axs[0].axis("on")

    im = axs[1].imshow(inverted, cmap='inferno')
    axs[1].set_title("Depth Map")
    axs[1].set_xlabel("Pixel X")
    axs[1].set_ylabel("Pixel Y")
    axs[1].axis("on")

    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Relative Depth")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def interpolate_zero_coords(coords):
    """Replace any (0, 0, 0) points in coords with the most recent non-zero point."""
    last_valid_valley = None
    for i in range(len(coords)):
        if np.array_equal(coords[i], [0, 0, 0]):
            if last_valid_valley is not None:
                coords[i] = last_valid_valley
        else:
            last_valid_valley = coords[i]

    # Forward-fill initial zero points
    for i in range(len(coords)):
        if np.array_equal(coords[i], [0, 0, 0]):
            coords[i] = last_valid_valley
        else:
            break

    return coords


def remove_outliers_zscore(data, threshold=2):
    """Remove outliers from a list based on the Z-score method."""
    z_scores = zscore(data)
    filtered_data = [val for val, z in zip(data, z_scores) if abs(z) < threshold]
    return filtered_data


def remove_outliers(data, threshold=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_indices = np.abs(data - mean) <= threshold * std_dev
    return filtered_indices


def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)


def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')


def ego_depth_estimator_cached(rgb_imgs, video_id, window_start, window_end, cache_dir, midas, midas_transform, device, SCALE_FACTOR):
    start_t_window = time.time()

    keypoint_json =  f"{video_id}_ego_resized_640x480_keypoints.json"
    print(f"Keypoint JSON: {keypoint_json}")
    cache_file = os.path.join(cache_dir, keypoint_json)

    cpr_depth = 0

    print(f"Loading cached keypoints for {video_id} from {cache_file}")
    with open(cache_file, 'r') as f:
        json_data = json.load(f)
    # JSON keys are strings, convert back to int
    keypoint_dict = {int(k): v for k, v in json_data.items()}

    print(f"Loaded {len(keypoint_dict)} keypoints from cache.")

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

    # scale wrist coordinates to match the depth map resolution (224, 224)
    # wrist_x = (wrist_x * 224 // 640).astype(int) # it seems only train set is 640x480. test set is 1920x1080
    # wrist_y = (wrist_y * 224 // 480).astype(int)

    wrist_x = (wrist_x *224 // 1920).astype(int)  # scale to 224x224
    wrist_y = (wrist_y *224 // 1080).astype(int)  # scale to 224x224

    # get the data within the start and end window
    wrist_x = wrist_x[window_start:window_end]
    wrist_y = wrist_y[window_start:window_end]
    start = window_start
    end = window_end
    print(f"Processing window from {start} to {end} for video {video_id}")
    print("Number of RGB images: ", len(rgb_imgs))
    print("Number of wrist keypoints: ", len(wrist_y))

    
    tensor_wrist_y = torch.tensor(wrist_y)
    try:
        low_pass_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
    except Exception as e:
        print(f"Error in low pass filter: {e}")
        return 0.0, 0.0, 0.0



    wrist_y_window = low_pass_wrist_y.numpy()
    wrist_x_window = wrist_x


    # Detect peaks/valleys (unfiltered)
    p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(wrist_y_window, mul=1, show=False)
    n_cpr_window_pred = (len(p) + len(v)) * 0.5

    # Filter outliers
    x_filtered_indices = remove_outliers(wrist_x_window, threshold=1)
    y_filtered_indices = remove_outliers(wrist_y_window, threshold=1)
    final_filtered_indices = x_filtered_indices & y_filtered_indices

    filtered_wrist_x = wrist_x_window[final_filtered_indices]
    filtered_wrist_y = wrist_y_window[final_filtered_indices]

    # Filter the images
    rgb_imgs_window = rgb_imgs
    rgb_imgs_window = np.array(rgb_imgs_window)
    filtered_rgb_imgs_window = rgb_imgs_window[final_filtered_indices]

    # Redetect peaks/valleys on filtered data
    p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(filtered_wrist_y, mul=1, show=False)
    wrist_x_p = filtered_wrist_x[p]
    wrist_y_p = filtered_wrist_y[p]
    wrist_x_v = filtered_wrist_x[v]
    wrist_y_v = filtered_wrist_y[v]

    print("len of filtered_rgb_imgs_window: ", len(filtered_rgb_imgs_window))
    print("Shape of one filtered_rgb_img: ", filtered_rgb_imgs_window[0].shape)

    depth_maps = []

    for i, img in enumerate(filtered_rgb_imgs_window):

        # Compute depth map for *all* frames, but only save if i is in peak or valley
        depth_map = estimate_depth_from_rgb(img, midas=midas, midas_transform=midas_transform, device=device)  # no midas debug figure
        depth_maps.append(depth_map)

    # Scale the depth maps
    abs_depth_maps = [dm * SCALE_FACTOR for dm in depth_maps]
    filtered_depth_imgs_window = np.array(abs_depth_maps)

    print(f"Number of depth maps: {len(filtered_depth_imgs_window)}")

    # Gather peak/valley depth maps
    filtered_depth_imgs_p = [filtered_depth_imgs_window[i] for i in p]
    filtered_depth_imgs_v = [filtered_depth_imgs_window[i] for i in v]

    print(f"Number of filtered depth maps for peaks: {len(filtered_depth_imgs_p)}")
    print(f"Number of filtered depth maps for valleys: {len(filtered_depth_imgs_v)}")


    peak_3d_points, valley_3d_points = [], []
    for i, depth_img in enumerate(filtered_depth_imgs_p):
        print(f"Wrist peak coordinates: ({wrist_x_p[i]}, {wrist_y_p[i]})  ")
        print(f"Depth image shape: {depth_img.shape}")
        if 0 <= wrist_x_p[i] < depth_img.shape[1] and 0 <= wrist_y_p[i] < depth_img.shape[0]:
            depth_val = depth_img[int(wrist_y_p[i]), int(wrist_x_p[i])]
            # print(f"Peak depth value at ({wrist_x_p[i]}, {wrist_y_p[i]}): {depth_val}")
            xyz = get_XYZ(wrist_x_p[i], wrist_y_p[i], depth_val, intrinsics)
            peak_3d_points.append(xyz)

    for i, depth_img in enumerate(filtered_depth_imgs_v):
        if 0 <= wrist_x_v[i] < depth_img.shape[1] and 0 <= wrist_y_v[i] < depth_img.shape[0]:
            depth_val = depth_img[int(wrist_y_v[i]), int(wrist_x_v[i])]
            xyz = get_XYZ(wrist_x_v[i], wrist_y_v[i], depth_val, intrinsics)
            valley_3d_points.append(xyz)

    print(f"Number of 3D peaks: {len(peak_3d_points)}")
    print(f"Number of 3D valleys: {len(valley_3d_points)}")


    peak_3d_points = interpolate_zero_coords(peak_3d_points)
    valley_3d_points = interpolate_zero_coords(valley_3d_points)

    distances = []
    for pk, vl in zip(peak_3d_points, valley_3d_points):
        dist = np.linalg.norm(pk - vl)
        distances.append(dist)

    distances = np.array(distances)
    distances = remove_outliers_zscore(distances, threshold=1.5)
    cpr_depth_3d = np.mean(distances) if len(distances) > 0 else 0
    print(f"Number of peaks: {len(peak_3d_points)}, Number of valleys: {len(valley_3d_points)}, distances: {len(distances)}, distances: {distances}")
    print(f"Predicted Mean distance between 3D peaks and valleys: {cpr_depth_3d:.2f} mm")

    # Compute relative depth difference (unscaled) for peaks vs valleys
    peak_depths = []
    valley_depths = []
    for i, (x, y) in enumerate(zip(wrist_x_p, wrist_y_p)):
        if 0 <= x < depth_maps[i].shape[1] and 0 <= y < depth_maps[i].shape[0]:
            peak_depths.append(depth_maps[i][int(y), int(x)])
    for i, (x, y) in enumerate(zip(wrist_x_v, wrist_y_v)):
        if 0 <= x < depth_maps[i].shape[1] and 0 <= y < depth_maps[i].shape[0]:
            valley_depths.append(depth_maps[i][int(y), int(x)])

    if len(peak_depths) > 0 and len(valley_depths) > 0:
        min_len = min(len(peak_depths), len(valley_depths))
        cpr_depth_rel = np.mean(np.array(peak_depths[:min_len]) - np.array(valley_depths[:min_len]))
    else:
        cpr_depth_rel = 0
    cpr_depth_rel = abs(cpr_depth_rel)
    abs_cpr_depth = cpr_depth_rel * SCALE_FACTOR
    print(f"Predicted CPR Depth (scaled, mm): {abs_cpr_depth:.2f}")

    # Ground Truth for the window
    # gt_window = gt_readings_for_clip
    # gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
    # n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

    # peak_depths_gt = gt_window[gt_peaks]
    # valley_depths_gt = gt_window[gt_valleys]
    # l = min(len(peak_depths_gt), len(valley_depths_gt))
    # gt_cpr_depth = float((peak_depths_gt[:l] - valley_depths_gt[:l]).mean()) if l > 0 else 0

    # print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")
    print(f"Predicted CPR Depth (mm): {cpr_depth_rel:.2f}")

    end_t_window = time.time()
    inference_time = end_t_window - start_t_window
    print(f"Time taken for window: {inference_time:.2f} seconds")

    log_msg = (
        f"Predicted_CPR_Depth:{cpr_depth_rel:.2f},"
        f"InferenceTimeSeconds:{inference_time:.4f},"
        f"3D_Pred_Depth:{cpr_depth_3d:.2f},"
        f"ABS_Pred_Depth:{abs_cpr_depth:.2f}"
    )
    print(log_msg)

    print("*" * 10, "End of window", "*" * 10)
    depth_maps.clear()
    del depth_maps

    return cpr_depth_rel, cpr_depth_3d, abs_cpr_depth


