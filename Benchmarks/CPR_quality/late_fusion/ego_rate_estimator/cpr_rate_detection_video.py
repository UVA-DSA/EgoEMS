import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
parent_directory = os.path.abspath('.')
print(parent_directory)
sys.path.append(parent_directory)
from tools import tools as depth_tools
import sys
import cv2
import torch
import re
import shutil

import time

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import numpy as np
import cv2
import argparse
import json
import os

import mediapipe as mp


from scipy.stats import zscore

sample_rate = 30
subsample_rate = 30
window_duration = 5  # 5-second window
window_frames = window_duration * sample_rate  # Number of frames per 5-second window

DEBUG = False


def draw_keypoints(frame, x_vals, y_vals):
    """Draw keypoints on the frame."""
    for x, y in zip(x_vals, y_vals):
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for keypoints
    return frame


def crop_img_bb(img, hand_bb, pad, show=False):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        raise Exception("Invalid image shape")
    hand_bb = [int(bb) for bb in hand_bb]
    img_crop = img[max(0, hand_bb[1] - pad):min(h, hand_bb[3] + pad), max(0, hand_bb[0] - pad):min(w, hand_bb[2] + pad)]
    if show:
        cv2.imshow("Image with Bounding Box", img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_crop


def get_bb(results):
    bbx_list = results.xyxy
    conf_list = results.confidence
    if len(conf_list) == 0:
        return []
    max_conf_arg = np.argmax(conf_list)
    bb = bbx_list[max_conf_arg]
    return bb


class WristDet_mediapipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.6
        )

    ### This function is used to get the keypoints of the closest hand wrist
    # def get_kypts(self, image):
    #     height, width, _ = image.shape
    #     results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     xy_vals = []
    #     z_vals = []
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             for i in range(21):
    #                 x = int(hand_landmarks.landmark[i].x * width)
    #                 y = int(hand_landmarks.landmark[i].y * height)
    #                 z = hand_landmarks.landmark[i].z
    #                 xy_vals.append((x, y))
    #                 z_vals.append(z)
    #         if len(xy_vals) == 42:
    #             closest_hand = np.argmin([z_vals[0], z_vals[21]])
    #             start_coord = 0 if closest_hand == 0 else 21
    #             xy_vals = xy_vals[start_coord:start_coord + 21]
    #     return image, xy_vals

    ### This function is used to get the keypoints of all hands
    def get_kypts(self, image):
        height, width, _ = image.shape
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        all_hands = []  # To store landmarks for all hands

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                xy_vals = []
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    xy_vals.append((x, y))
                all_hands.append(xy_vals)  # Add current hand's landmarks to the list

        return image, all_hands



### This function is used to get the keypoints of all hands
def get_kpts(img, wrst, base_model, frame_num, video_id):
    results = base_model.predict(img)
    bb = get_bb(results)

    if bb is None or len(bb) == 0:  # Check if bounding box is empty
        print("No bounding box detected.")
        return {"hands": []}  # Return empty list for hands
    
    pad = 80
    img_crop = crop_img_bb(img, bb, pad, show=False)
    # save the cropped image as debug with a unique name for each frame
    cv2.imwrite(f"./ego_rate_estimator/debug/cropped_{video_id}_{frame_num}.jpg", img_crop)


    image, all_hands = wrst.get_kypts(img_crop)


    hands_data = []
    for hand in all_hands:
        x_vals = [int(val[0] + bb[0] - pad) for val in hand]
        y_vals = [int(val[1] + bb[1] - pad) for val in hand]
        hands_data.append({"x": x_vals, "y": y_vals})

    # visualize the cropped image with keypoints
    if len(hands_data) > 0:
        for hand in hands_data:
            img = draw_keypoints(img, hand["x"], hand["y"])

        # save the cropped image as debug with a unique name for each frame
        save_path = f"./ego_rate_estimator/debug/wrist_keypoints/{video_id}_{frame_num}.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"Saved debug image: {save_path}")

    return {"hands": hands_data}


# def get_kpts_ego(img, wrst, base_model, frame_num, video_id):
#     """
#     1) DINO detects ALL hands → base_model.predict(img)
#     2) pick the box whose CENTER is closest to frame center
#     3) crop+pad → hand_crop
#     4) wrst.get_kypts(hand_crop) → (crop_img, all_hands)
#     5) take the first hand_kps = all_hands[0], map its 21 (x,y)s back.
#     """
#     H, W = img.shape[:2]

#     # 1) DINO hand detections
#     results = base_model.predict(img)
#     boxes = results.xyxy       # list of [x1,y1,x2,y2]
#     if len(boxes) == 0:
#         return {"hands": []}

#     # 2) pick the box nearest the image center
#     cx0, cy0 = W/2.0, H/2.0
#     dists = [ ((b[0]+b[2])/2 - cx0)**2 + ((b[1]+b[3])/2 - cy0)**2 for b in boxes ]
#     best = int(np.argmin(dists))
#     x1, y1, x2, y2 = [int(v) for v in boxes[best]]

#     # 3) crop + pad
#     pad = 40
#     x1p = max(0, x1 - pad)
#     y1p = max(0, y1 - pad)
#     x2p = min(W, x2 + pad)
#     y2p = min(H, y2 + pad)
#     crop = img[y1p:y2p, x1p:x2p]

#     # optional debug dump
#     dbg = "./ego_rate_estimator/debug"
#     os.makedirs(dbg, exist_ok=True)
#     cv2.imwrite(f"{dbg}/crop_{video_id}_{frame_num}.jpg", crop)

#     # 4) Mediapipe on that crop
#     _, all_hands = wrst.get_kypts(crop)   # unpack the two‑item return
#     if not all_hands:
#         print("No hands detected in the cropped image.")
#         return {"hands": []}

#     # 5) take first (largest) hand's 21 keypoints
#     hand_kps = all_hands[0]  # list of 21 (x,y)

#     # map them back
#     mapped_x, mapped_y = [], []
#     for (cx, cy) in hand_kps:
#         mapped_x.append(cx + x1p)
#         mapped_y.append(cy + y1p)

#     # print(f"Frame {frame_num}: Keypoints X: {mapped_x}")
#     # print(f"Frame {frame_num}: Keypoints Y: {mapped_y}")

#     return {"hands": [{"x": mapped_x, "y": mapped_y}]}

def get_kpts_ego(img, wrst, base_model, frame_num, video_id):
    """
    1) DINO detects ALL hands → base_model.predict(img)
    2) if no boxes: return empty
    3) run wrst.get_kypts on the full image
    4) take the first hand's 21 (x,y)s and return
    """
    results = base_model.predict(img)
    bb = get_bb(results)

    if bb is None or len(bb) == 0:  # Check if bounding box is empty
        print("No bounding box detected.")
        return {"hands": []}  # Return empty list for hands
    
    pad = 80
    img_crop = crop_img_bb(img, bb, pad, show=False)
    # save the cropped image as debug with a unique name for each frame
    if DEBUG:
        cv2.imwrite(f"./ego_rate_estimator/debug/cropped_{video_id}_{frame_num}.jpg", img_crop)


    image, all_hands = wrst.get_kypts(img_crop)


    hands_data = []
    x_vals = []
    y_vals = []
    for hand in all_hands:
        x_vals = [int(val[0] + bb[0] - pad) for val in hand]
        y_vals = [int(val[1] + bb[1] - pad) for val in hand]
        hands_data.append({"x": x_vals, "y": y_vals})

    # visualize the cropped image with keypoints
    # if len(hands_data) > 0:
    #     for hand in hands_data:
    #         img = draw_keypoints(img, hand["x"], hand["y"])

        # save the cropped image as debug with a unique name for each frame
        # save_path = f"./ego_rate_estimator/debug/wrist_keypoints/{video_id}_{frame_num}.jpg"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imwrite(save_path, img)
        # print(f"Saved debug image: {save_path}")

    return {"hands": [{"x": x_vals, "y": y_vals}]}




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

def visualize_wrist_keypoints(wrist_x, wrist_y, video_frames, output_path):
    # Create a copy of the video frames to draw on
    frames_copy = [frame.copy() for frame in video_frames
                   if frame.shape[0] > 0 and frame.shape[1] > 0]
    for i, (x, y) in enumerate(zip(wrist_x, wrist_y)):  
        if x > 0 and y > 0 and i < len(frames_copy):
            # Draw a circle on the frame
            cv2.circle(frames_copy[i], (int(x), int(y)), 5, (255, 0, 0), -1)
    # Save the frames as individually with frame_numbers
    os.makedirs(output_path, exist_ok=True)

    for i, frame in enumerate(frames_copy):
        cv2.imwrite(f"{output_path}/frame_{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    # video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    # for frame in frames_copy:
    #     video_writer.write(frame)
    # video_writer.release()




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


def pick_first_valid(x_vals, y_vals):
    """
    Return the first (x,y) where both >0.
    If none are valid, return (None, None).
    """
    for x, y in zip(x_vals, y_vals):
        if x > 0 and y > 0:
            return x, y
    return None, None


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












# Define paths
# Windows environment
# GT_path = r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
# data_path = r'D:\EgoExoEMS_CVPR2025\CPR Test\GoPro_CPR_Clips\ego_gopro_cpr_clips\test_root'
# log_path = rf"E:\EgoExoEMS\Benchmarks\CPR_quality\vision\results\unsupervised_ego_rate_window_test_split_results.txt"
# debug_plots_path = rf"E:\EgoExoEMS\Benchmarks\CPR_quality\vision\unsupervised_ego_rate_window_debug_plots"

if __name__ == "__main__":

    # Linux environment
    BASE_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025"  # Set an environment variable or update this
    BASE_REPO_DIR = "/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS"
    GT_path = os.path.join(BASE_DIR, "Dataset", "Final")
    data_path = os.path.join(BASE_DIR, "Dataset", "GoPro_CPR_Clips", "ego_gopro_cpr_clips", "test_root")

    if DEBUG:
        log_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", "debug", f"debug_unsupervised_ego_rate_window_test_split_results.txt")
    else:
        log_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", "results", f"unsupervised_ego_rate_window_test_split_results.txt")

    debug_plots_path = os.path.join(BASE_REPO_DIR, "Benchmarks", "CPR_quality", "vision", "debug", f"unsupervised_ego_rate_window_debug_plots")

    # delete the directory if it already exists
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

        all_filtered_wrist_x, all_filtered_wrist_y = [], []

        print("number of wrist keypoints: ", len(low_pass_wrist_y))
        for start in range(0, len(low_pass_wrist_y), window_frames):
            end = start + window_frames
            print("Window: ", start, end, len(low_pass_wrist_y[start:end]))
            if end > len(low_pass_wrist_y):
                break  # Stop if the last window is incomplete

            # Predicted CPR cycles for the window
            wrist_y_window = low_pass_wrist_y[start:end].numpy()
            wrist_x_window = wrist_x[start:end]

            start_t = time.time()

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


            print(f"Predicted CPR rate per window: {n_cpr_window_pred} cycles")
            print(f"Ground truth CPR rate per window: {n_cpr_window_gt} cycles")

            predicted_CPR_rate = n_cpr_window_pred * 60 / window_duration
            ground_truth_CPR_rate = n_cpr_window_gt * 60 / window_duration

            print(f"Predicted CPR rate (BPM): {predicted_CPR_rate}")
            print(f"Ground truth CPR rate (BPM): {ground_truth_CPR_rate}")

            end_t = time.time()
            inference_time = end_t - start_t
            print(f"Time taken for window: {inference_time} seconds")

            # Visualize wrist keypoints on the filtered RGB images
            if DEBUG:
                visualize_wrist_keypoints(wrist_x_p, wrist_y_p, filtered_rgb_imgs_p, f'{debug_plots_path}/{json_file.split(".")[0]}_window_{window_num}_wrist_kps')

            log_msg = (f"File:{json_file},Window:{window_num},Predicted_CPR_Rate:{predicted_CPR_rate},GT_CPR_Rate:{ground_truth_CPR_rate},InferenceTimeSeconds:{inference_time}")
            write_log_line(log_path, log_msg)
            print(log_msg)



        # Clear large arrays after processing
        rgb_imgs.clear()

        # Explicitly delete variables with large data
        del rgb_imgs

        
base_model = None
wrst = None

# Initialize models
def init_models():
    global base_model, wrst
    # Initialize the model
    base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))

    # Initialize the wrist detector
    wrst = WristDet_mediapipe()

    print("Models initialized.")

init_models()

def ego_rate_detect(rgb_imgs,video_id):

    # convert rgb_imgs to numpy array
    if isinstance(rgb_imgs, torch.Tensor):
        rgb_imgs = rgb_imgs.cpu().numpy()
        rgb_imgs = np.array(rgb_imgs, dtype=np.uint8)
        rgb_imgs = np.transpose(rgb_imgs, (0, 2, 3, 1))
        rgb_imgs = np.array(rgb_imgs, dtype=np.uint8)

    print("Starting ego rate detection...")
    print("Number of RGB images: ", rgb_imgs.shape)

    # print("base_model: ", base_model)
    # print("wrst: ", wrst)

    # Process each frame in the window rgb_imgs
    keypoint_dict = {}
    for i, img in enumerate(rgb_imgs):
        # print("Processing frame: ", i)
        # print("Image shape: ", img.shape)
        # print("Image type: ", type(img))
        # print("Image dtype: ", img.dtype)

        # make image is open cv format from tensor
        # resize the image to 640x480
        img = cv2.resize(img, (640, 480))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get keypoints
        kpts = get_kpts_ego(img, wrst, base_model, i, video_id)

        # visualize the keypoints on the image
        if DEBUG:
            try:
                img_with_kpts = draw_keypoints(img, kpts['hands'][0]['x'], kpts['hands'][0]['y'])
                cv2.imwrite(f"./ego_rate_estimator/debug/wrist_kp_{video_id}_{i}.jpg", img_with_kpts)
            except Exception as e:
                print(f"Error visualizing keypoints for frame {i}: {e}")
                continue
        keypoint_dict[i] = kpts

    wrist_x, wrist_y = [], []
    # Extract wrist coordinates from keypoint_dict
    for i in range(len(rgb_imgs)):
        try:
            if keypoint_dict[i]['hands']:
                xs = keypoint_dict[i]['hands'][0]['x'][0]
                ys = keypoint_dict[i]['hands'][0]['y'][0]
                # x_i, y_i = pick_first_valid(xs, ys)
                # if x_i is None:
                #     continue
                wrist_x.append(xs)
                wrist_y.append(ys)
        except Exception as e:
            print(f"Wrist IndexError for frame {i}: {e}")
            continue



    # Convert wrist coordinates to NumPy arrays for easier manipulation
    wrist_y = np.array(wrist_y, dtype=float)
    wrist_x = np.array(wrist_x, dtype=float)

    # Use the matching_frame_indices to retrieve the corresponding RGB and depth frames
    matching_rgb_frames = [rgb_imgs[i] for i in range(len(rgb_imgs)) if i in keypoint_dict]
    matching_rgb_frames = np.array(matching_rgb_frames)
    print("Number of matching RGB frames: ", len(matching_rgb_frames))
    print("Number of wrist keypoints: ", len(wrist_y))

    # apply low pass filter to the wrist y coordinates
    tensor_wrist_y = torch.tensor(wrist_y)
    try:
        low_pass_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
    except Exception as e:
        print(f"Error in low pass filter: {e}")
        return None
    
    wrist_y = low_pass_wrist_y.numpy()
                    # Filter indices based on outlier detection for both X and Y
    x_filtered_indices = remove_outliers(wrist_x, threshold=1)
    y_filtered_indices = remove_outliers(wrist_y, threshold=1)

    # Combine filters to maintain synchronization
    final_filtered_indices = x_filtered_indices & y_filtered_indices
    # print("Final filtered indices: ", final_filtered_indices, len(final_filtered_indices))

    # Filter the arrays
    filtered_wrist_x = wrist_x[final_filtered_indices]
    filtered_wrist_y = wrist_y[final_filtered_indices]

    # Output the filtered data
    # print("Filtered Wrist X:", filtered_wrist_x)
    # print("Filtered Wrist Y:", filtered_wrist_y)

    # filter the depth images for the window


    print("number of filtered wrist keypoints: ", len(filtered_wrist_x))

    # Filter indices based on outlier detection for both X and Y
    p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(filtered_wrist_y, mul=1, show=False)
    n_cpr_window_pred = (len(p) + len(v)) * 0.5

    print(f"Predicted CPR rate per window: {n_cpr_window_pred} cycles")
    predicted_CPR_rate = n_cpr_window_pred * 60 / window_duration

    print(f"Predicted CPR rate (BPM): {predicted_CPR_rate}")

    return predicted_CPR_rate


def ego_rate_detect_cached(rgb_imgs, video_id, window_start, window_end, cache_dir):
    """
    For a given video (as list/array of RGB frames) and its ID,
    either load cached keypoints or extract them, then compute CPR rate.
    """

    keypoint_json =  f"{video_id}_ego_resized_640x480_keypoints.json"
    print(f"Keypoint JSON: {keypoint_json}")
    cache_file = os.path.join(cache_dir, keypoint_json)



    rate_bpm = None

    # 1) Load or extract keypoints
    if os.path.exists(cache_file):
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
            return None


        print("number of wrist keypoints: ", len(low_pass_wrist_y))
    
        # Predicted CPR cycles for the window

        wrist_y_window = low_pass_wrist_y.numpy()
        wrist_x_window = wrist_x

        start_t = time.time()

                    # Filter indices based on outlier detection for both X and Y
        x_filtered_indices = remove_outliers(wrist_x_window, threshold=1)
        y_filtered_indices = remove_outliers(wrist_y_window, threshold=1)

        # Combine filters to maintain synchronization
        final_filtered_indices = x_filtered_indices & y_filtered_indices
        # print("Final filtered indices: ", final_filtered_indices, len(final_filtered_indices))

        # Filter the arrays
        filtered_wrist_x = wrist_x_window[final_filtered_indices]
        filtered_wrist_y = wrist_y_window[final_filtered_indices]

        # filter the depth images for the window
        rgb_imgs_window = rgb_imgs[:150]
        rgb_imgs_window = np.array(rgb_imgs_window)



        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(wrist_y_window, mul=1, show=False)
        n_cpr_window_pred = (len(p) + len(v)) * 0.5


        print("number of filtered wrist keypoints: ", len(filtered_wrist_x))

        # Filter indices based on outlier detection for both X and Y

        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(filtered_wrist_y, mul=1, show=False)

        print(f"Predicted CPR rate per window: {n_cpr_window_pred} cycles")

        predicted_CPR_rate = n_cpr_window_pred * 60 / window_duration

        print(f"Predicted CPR rate (BPM): {predicted_CPR_rate}")

        rate_bpm = predicted_CPR_rate

        end_t = time.time()
        inference_time = end_t - start_t
        print(f"Time taken for window: {inference_time} seconds")

    else:
        print(f"No cache found for {video_id}. ")
        return None


    print(f"Predicted CPR rate (BPM): {rate_bpm}")
    return rate_bpm