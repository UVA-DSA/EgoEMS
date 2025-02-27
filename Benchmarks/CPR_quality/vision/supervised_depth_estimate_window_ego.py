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
from Tools.depth_sensor_processing import tools as depth_tools

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


# get command line argument for model type using argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="DPT_Large", help="DPT_Large, DPT_Hybrid, MiDaS_small")
parser.add_argument("--slurm_id", type=str, default="none",
                    help="Optional: Slurm job ID for logging.")

args = parser.parse_args()
model_type = args.model_type
slurm_id = args.slurm_id    

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


def estimate_depth_from_rgb(rgb_img, output_path=None):
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


# Linux environment paths
BASE_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025"  # Adjust as needed
BASE_REPO_DIR = "/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS"
GT_path = os.path.join(BASE_DIR, "Dataset", "Final")
data_path = os.path.join(BASE_DIR, "Dataset", "GoPro_CPR_Clips", "ego_gopro_cpr_clips", "test_root")
log_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", "results",
                        f"job_{slurm_id}_supervised_{model_type}_ego_depth_window_test_split_results.txt")
debug_plots_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", "debug",
                                f"supervised_{model_type}_ego_depth_window_debug_plots")

if os.path.exists(debug_plots_path):
    shutil.rmtree(debug_plots_path)
os.makedirs(debug_plots_path, exist_ok=True)

init_log(log_path)

data_dir = os.path.join(data_path, 'chest_compressions')
json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
mp4_files = [file for file in os.listdir(data_dir) if file.endswith('.mp4')]

for json_file in json_files:
    json_path = os.path.join(data_dir, json_file)
    json_data = json.load(open(json_path))
    matching_mp4 = [f for f in mp4_files if json_file.replace('_keypoints.json', '') in f]
    if not matching_mp4:
        continue
    mp4_file = matching_mp4[0]

    if "P20_ts5_ks4_0_keypoints" not in json_file:
        continue
    
    print("*" * 20)
    print(f"Processing file: {json_file}")
    print(f"Processing video: {mp4_file}")


    rgb_imgs_full = read_video(os.path.join(data_dir, mp4_file))
    if len(rgb_imgs_full) != len(json_data.keys()):
        # If mismatch, skip
        continue

    # Sort keys
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
    rgb_imgs = [rgb_imgs_full[i] for i in matching_frame_indices]

    # Low-pass filter
    tensor_wrist_y = torch.tensor(wrist_y)
    try:
        low_pass_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
    except Exception as e:
        print(f"Error in low pass filter: {e}")
        continue

    print(f"Number of wrist keypoints: {len(low_pass_wrist_y)}")

    # Get start/end frames from filename (if possible)
    match = re.search(r"(\d+\.\d+)_(\d+\.\d+)_ego\.mp4", mp4_file)
    start_frame = end_frame = None
    if match:
        start_t = float(match.group(1))
        end_t = float(match.group(2))
        start_frame = int(start_t * sample_rate)
        end_frame = int(end_t * sample_rate)

    # Load ground truth
    sbj = json_file.split('_')[0]
    t = json_file.split('_')[1][1:]
    gt_path = os.path.join(GT_path, sbj, 'cardiac_arrest', t, 'distance_sensor_data', 'sync_depth_sensor.csv')
    with open(gt_path, 'r') as file:
        gt_lines = file.readlines()
    gt_lines = np.array([float(line.strip()) for line in gt_lines[1:]])
    gt_readings_for_clip = gt_lines[start_frame:end_frame]

    # Create a subfolder for the current JSON file (debug frames)
    file_debug_folder = os.path.join(debug_plots_path, json_file.replace('.json', ''))
    os.makedirs(file_debug_folder, exist_ok=True)

    for start_idx in range(0, len(low_pass_wrist_y), window_frames):
        end_idx = start_idx + window_frames
        if end_idx > len(low_pass_wrist_y):
            break

        start_t_window = time.time()

        wrist_y_window = low_pass_wrist_y[start_idx:end_idx].numpy()
        wrist_x_window = wrist_x[start_idx:end_idx]

        print("*" * 20)
        window_num = start_idx // window_frames + 1
        print(f"Processing window {window_num} for {json_file}")
        print("Processing window frames: ", start_idx, end_idx, "Window length:", len(wrist_y_window))

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
        rgb_imgs_window = rgb_imgs[start_idx:end_idx]
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
        global_indices_window = matching_frame_indices[start_idx:end_idx]
        filtered_global_indices = [
            global_indices_window[i]
            for i in range(len(global_indices_window))
            if final_filtered_indices[i]
        ]

        # --- CHANGED: We'll only save frames for peaks/valleys. Let's collect them in a set:
        peak_set = set(p)      # indices within the filtered data that are peaks
        valley_set = set(v)    # indices within the filtered data that are valleys

        for i, img in enumerate(filtered_rgb_imgs_window):
            frame_idx = filtered_global_indices[i]

            # Compute depth map for *all* frames, but only save if i is in peak or valley
            depth_map = estimate_depth_from_rgb(img, output_path=None)  # no midas debug figure
            depth_maps.append(depth_map)

            # Only save if this is a peak or valley index
            if i in peak_set or i in valley_set:  # <-- CHANGED
                if DEBUG:
                    # Build paths
                    depth_legend_path = os.path.join(file_debug_folder, f"depth_legend_frame_{frame_idx:06d}.png")
                    debug_frame_path = os.path.join(file_debug_folder, f"depth_frame_{frame_idx:06d}.png")
                    debug_rgb_frame_path = os.path.join(file_debug_folder, f"rgb_frame_{frame_idx:06d}.png")
                    combined_fig_path = os.path.join(file_debug_folder, f"combined_frame_side-by-side{frame_idx:06d}.png")
                    combined_top_bottom_fig_path = os.path.join(file_debug_folder, f"combined_frame_top-bottom{frame_idx:06d}.png")

                    # 1) Save "legend" figure
                    _ = estimate_depth_from_rgb(img, output_path=depth_legend_path)

                    # Convert depth map to a colormap
                    dmin, dmax = depth_map.min(), depth_map.max()
                    if dmax - dmin < 1e-6:
                        dmax = dmin + 1e-6
                    normalized = (depth_map - dmin) / (dmax - dmin)
                    inverted = (1.0 - normalized) * 255.0
                    depth_8u = inverted.astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

                    wx = int(filtered_wrist_x[i])
                    wy = int(filtered_wrist_y[i])
                    if 0 <= wx < depth_color.shape[1] and 0 <= wy < depth_color.shape[0]:
                        cv2.circle(depth_color, (wx, wy), 5, (0, 255, 0), -1)

                    rgb_annotated = img.copy()
                    if 0 <= wx < rgb_annotated.shape[1] and 0 <= wy < rgb_annotated.shape[0]:
                        cv2.circle(rgb_annotated, (wx, wy), 5, (0, 255, 0), -1)

                    cv2.imwrite(debug_frame_path, depth_color)
                    cv2.imwrite(debug_rgb_frame_path, cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR))

                    # Save side-by-side figure
                    save_side_by_side_figure(
                        rgb_annotated,
                        depth_map,
                        output_path=combined_fig_path,
                        wrist_xy=(wx, wy)
                    )
                    # Save top-bottom figure
                    save_top_bottom_figure(
                        rgb_annotated,
                        depth_map,
                        output_path=combined_top_bottom_fig_path,
                        wrist_xy=(wx, wy)
                    )
        # --- END CHANGED

        # Scale the depth maps
        abs_depth_maps = [dm * SCALE_FACTOR for dm in depth_maps]
        filtered_depth_imgs_window = np.array(abs_depth_maps)

        # Gather peak/valley depth maps
        filtered_depth_imgs_p = [filtered_depth_imgs_window[i] for i in p]
        filtered_depth_imgs_v = [filtered_depth_imgs_window[i] for i in v]

        peak_3d_points, valley_3d_points = [], []
        for i, depth_img in enumerate(filtered_depth_imgs_p):
            if 0 <= wrist_x_p[i] < depth_img.shape[1] and 0 <= wrist_y_p[i] < depth_img.shape[0]:
                depth_val = depth_img[int(wrist_y_p[i]), int(wrist_x_p[i])]
                xyz = get_XYZ(wrist_x_p[i], wrist_y_p[i], depth_val, intrinsics)
                peak_3d_points.append(xyz)

        for i, depth_img in enumerate(filtered_depth_imgs_v):
            if 0 <= wrist_x_v[i] < depth_img.shape[1] and 0 <= wrist_y_v[i] < depth_img.shape[0]:
                depth_val = depth_img[int(wrist_y_v[i]), int(wrist_x_v[i])]
                xyz = get_XYZ(wrist_x_v[i], wrist_y_v[i], depth_val, intrinsics)
                valley_3d_points.append(xyz)

        peak_3d_points = interpolate_zero_coords(peak_3d_points)
        valley_3d_points = interpolate_zero_coords(valley_3d_points)

        distances = []
        for pk, vl in zip(peak_3d_points, valley_3d_points):
            dist = np.linalg.norm(pk - vl)
            distances.append(dist)

        distances = np.array(distances)
        distances = remove_outliers_zscore(distances, threshold=1.5)
        cpr_depth_3d = np.mean(distances) if len(distances) > 0 else 0
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
        gt_window = gt_readings_for_clip[start_idx:end_idx]
        gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
        n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

        peak_depths_gt = gt_window[gt_peaks]
        valley_depths_gt = gt_window[gt_valleys]
        l = min(len(peak_depths_gt), len(valley_depths_gt))
        gt_cpr_depth = float((peak_depths_gt[:l] - valley_depths_gt[:l]).mean()) if l > 0 else 0

        print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")
        print(f"Predicted CPR Depth (mm): {cpr_depth_rel:.2f}")

        end_t_window = time.time()
        inference_time = end_t_window - start_t_window
        print(f"Time taken for window: {inference_time:.2f} seconds")

        log_msg = (
            f"File:{json_file},Window:{window_num},"
            f"Predicted_CPR_Depth:{cpr_depth_rel:.2f},"
            f"GT_CPR_Depth:{gt_cpr_depth:.2f},"
            f"InferenceTimeSeconds:{inference_time:.4f},"
            f"3D_Pred_Depth:{cpr_depth_3d:.2f},"
            f"ABS_Pred_Depth:{abs_cpr_depth:.2f}"
        )
        write_log_line(log_path, log_msg)
        print(log_msg)

        print("*" * 10, "End of window", "*" * 10)
        depth_maps.clear()
        del depth_maps

    print("*" * 20)
    rgb_imgs.clear()
    del rgb_imgs

print("[INFO] Done processing all files.")
print("[INFO] Results logged to: ", log_path)
print("[INFO] Debug plots saved to: ", debug_plots_path)
print("-" * 20)
