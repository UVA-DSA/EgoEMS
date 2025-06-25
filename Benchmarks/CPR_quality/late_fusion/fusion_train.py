#!/usr/bin/env python3

"""
Train a linear fusion model to optimally combine
smartwatch and ego-video CPR rate estimates.
Usage:
    python fusion_train.py 
        --annotation_file Annotations/main_annotation_cpr_quality.json 
        --data_base_path /path/to/ego_exo_data 
        --split split_1 
        --window_size 150 
        --batch_size 1
"""

import argparse
import os

import pandas as pd


from utils.utils import *
from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models
from datautils.ems import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from sklearn.linear_model import LinearRegression

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import argparse
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


from utils.utils import preprocess, detect_rate
from ego_depth_estimator.supervised_depth_estimate_window_ego import initialize_midas_model
from smartwatch_depth_estimator.cpr_depth_detection_smartwatch import initialize_smartwatch_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cache dir for ego video hand keypoints
CACHE_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/GoPro_CPR_Clips/ego_gopro_cpr_clips/train_root/chest_compressions/"


# def detect_rate(frames, smartwatch, gt, window_size, video_id):
#     """
#     Slide a tumbling window and compute:
#       - smartwatch_rate per window
#       - ego_video_rate per window
#       - gt_rate per window
#     Returns three 1D numpy arrays: (sw_rates, vid_rates, gt_rates)
#     """
#     sw_rates, vid_rates, gt_rates = [], [], []
#     total = frames.shape[0]

#     print(f"Processing video {video_id} with {total} frames, smartwatch data length {len(smartwatch)}, and gt data length {len(gt)}")

#     for start in range(0, total, window_size):
#         end = start + window_size
#         if end > total:
#             break

#         print(f"Processing window {start}:{end} for video {video_id}...")

#         try:
                
#             win_frames = frames[start:end]
#             win_smart = smartwatch[start:end]
#             win_gt = gt[start:end]

#             print(f"Window size: {len(win_frames)} frames, {len(win_smart)} smartwatch data points, {len(win_gt)} GT data points")
# # 
#             # v_rate = ego_rate_detect(win_frames, video_id)
#             v_rate = ego_rate_detect_cached(win_frames, video_id, start, end, CACHE_DIR)
#             s_rate = smartwatch_rate_detect(win_smart)
#             g_rate = get_gt_cpr_rate(win_gt)

#             print(f"Rates for window {start}:{end} - Video: {v_rate}, Smartwatch: {s_rate}, GT: {g_rate}")
#             # skip if any rate is None

#             # ensure Python floats
#             try:
#                 v_rate = float(v_rate)
#                 s_rate = float(s_rate)
#                 g_rate = float(g_rate)
#             except (TypeError, ValueError):
#                 continue

#             # skip NaNs
#             if np.isnan(v_rate) or np.isnan(s_rate) or np.isnan(g_rate):
#                 continue

#             sw_rates.append(s_rate)
#             vid_rates.append(v_rate)
#             gt_rates.append(g_rate)

#             print(f"window:{start}:{end},sw_rate:{s_rate:.2f},ego_rate:{v_rate:.2f},gt_rate:{g_rate:.2f}")

#         except Exception as e:
#             print(f"Error processing window {start}:{end} for video {video_id}: {e}")
#             # stack trace for debugging
#             import traceback
#             traceback.print_exc()
#             continue
#     return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=150,
                        help="Frames per window (e.g. 5s @30Hz =150)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--job_id', type=str, default="default")
    args = parser.parse_args()
    job = args.job_id
    print(f"Job ID: {job}")

    # prepare dataset & loader
    args = DefaultArgsNamespace()

    keysteps = args.dataloader_params['keysteps']
    out_classes = len(keysteps)

    modality = args.dataloader_params['modality']
    print("Modality: ", modality)
    print("Num of classes: ", out_classes)

    window = args.dataloader_params['observation_window']
    print("Window: ", window)

    task = args.dataloader_params['task']
    print("Task: ", task)

    train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)
    args.dataloader_params['train_class_stats'] = train_class_stats
    args.dataloader_params['val_class_stats'] = val_class_stats


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get current date and time
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create results directory if it doesn't exist
    results_dir = f"./results/fusion_train_{date_time_str}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    else:
        print(f"Results directory already exists: {results_dir}")

    debug_path = f"./ego_rate_estimator/debug/{date_time_str}"


    # create a CSV file path
    csv_path = f"{results_dir}/{job}_train_fusion_rates_results.csv"

    # remove if already exists
    if os.path.exists(csv_path):
        os.remove(csv_path)


    # define variable for model_size
    model_type="DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type="DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type="MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # initializing midas model for depth estimation
    midas_model, midas_transform, SCALE_FACTOR = initialize_midas_model(model_type=model_type, device=device)

    # initialize smartwatch models
    smartwatch_model, smartwatch_optimizer, smartwatch_criterion, smartwatch_scheduler = initialize_smartwatch_model(device=device)


    debug_csv = f"{results_dir}/debug_{job}_train_fusion_debug_per_batch.csv"
    debug_df = pd.DataFrame(columns=["subject_id", "trial_id", "start_idx", "smartwatch_rates", "video_rates", "gt_rates", "fused_rates"])
    debug_df.to_csv(debug_csv, mode='a', header=False, index=False)

    # remove the folder
    if os.path.exists(debug_path):
        # remove the folder and its contents
        print(f"Removing existing debug folder: {debug_path}")
        if os.path.isfile(debug_path):
            os.remove(debug_path)

    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # loop for few epochs
    print("Starting training...")
    for epoch in range(5):  # just one epoch for now

        # accumulate per-window rates across all subjects
        X_list, Y_list = [], []
        X_depth_list, Y_depth_list = [], []

        for batch in train_loader:

            print("**" * 20)
            print("--" * 20)

            subj = batch['subject_id'][0]
            trial = batch['trial_id'][0]
            start_t = batch['start_t'][0]
            end_t = batch['end_t'][0]
            keystep_id = batch['keystep_id'][0]

            # convert tensor to string for video_id
            print(f"Processing Subject: {subj}, Trial: {trial}, Keystep_ID: {keystep_id}, Start: {start_t}, End: {end_t}")
    
            start_t = float(start_t)
            end_t = float(end_t)
            start_t = f"{start_t:.3f}"
            end_t = f"{end_t:.3f}"
            video_id = f"{subj}_t{trial}_ks{keystep_id}_{start_t}_{end_t}"

            print(f"Video ID: {video_id}")

            input, feature_size, gt_sensor_data = preprocess(batch, modality, None, device, task=task)


            frames = input['frames'][0]    # [T,3,H,W]
            smartwatch = input['smartwatch'][0]    # [T,3]
            gt_sensor_data = gt_sensor_data[0]     # [T,3]

            if torch.is_tensor(frames):
                frames = frames.permute(0,2,3,1).cpu().numpy().astype(np.uint8)

            # move smartwatch to gpu
            
            sw_rates, vid_rates, gt_rates = detect_rate(frames, smartwatch, gt_sensor_data, window, video_id, CACHE_DIR)
            print(f"Smartwatch rates: {sw_rates}")
            print(f"Ego-video rates: {vid_rates}")
            print(f"GT rates: {gt_rates}")

            sw_depths, vid_depths, gt_depths = detect_depth(frames, smartwatch, gt_sensor_data, window, video_id, CACHE_DIR, midas_model, midas_transform, device, SCALE_FACTOR,  smartwatch_model, smartwatch_optimizer, smartwatch_criterion)

            print(f"Smartwatch depths: {sw_depths}")
            print(f"Ego-video depths: {vid_depths}")
            print(f"GT depths: {gt_depths}")

            if sw_rates.size == 0 or vid_rates.size == 0 or gt_rates.size == 0:
                print(f"{subj}_{trial}: no valid windows, skipping.")
                continue
            
            # if any of the depths are empty, skip this batch
            if sw_depths.size == 0 or vid_depths.size == 0 or gt_depths.size == 0:
                print(f"{subj}_{trial}: no valid depths, skipping.")
                continue
                

            # trim leading windows until we hit the modal GT rate
            sw_rates, vid_rates, gt_rates, start_idx = trim_pre_cpr(sw_rates, vid_rates, gt_rates)
            # print(f"Trimmed Smartwatch rates: {sw_rates}")
            # print(f"Trimmed Video rates: {vid_rates}")
            # print(f"Trimmed GT rates: {gt_rates}")



            # average the rates over the window
            s_rate = np.mean(sw_rates)
            v_rate = np.mean(vid_rates)
            g_rate = np.mean(gt_rates)

            # average the depths over the window
            s_depth = np.mean(sw_depths)
            v_depth = np.mean(vid_depths)
            g_depth = np.mean(gt_depths)


            # Create a row and save to CSV immediately
            row = pd.DataFrame([{
                "subject_id": batch['subject_id'],
                "trial_id": batch['trial_id'],
                "smartwatch_rate": s_rate,
                "video_rate": v_rate,
                "gt_rate": g_rate,
                "smartwatch_depth": s_depth,
                "video_depth": v_depth,
                "gt_depth": g_depth,
            }])

            debug_row = {
                "subject_id": subj,
                "trial_id": trial,
                "start_idx": start_idx,
                "smartwatch_rates": sw_rates.tolist(),
                "video_rates": vid_rates.tolist(),
                "gt_rates": gt_rates.tolist(),
                "smartwatch_depths": sw_depths.tolist(),
                "video_depths": vid_depths.tolist(),
                "gt_depths": gt_depths.tolist()
                                }
            
            debug_df = pd.DataFrame([debug_row])
            debug_df.to_csv(debug_csv, mode='a', header=False, index=False)


            # Write to CSV, with header only on first write
            write_header = not os.path.exists(csv_path)
            row.to_csv(csv_path, mode='a', index=False, header=write_header)

            Xb = np.stack([sw_rates, vid_rates], axis=1)
            X_list.append(Xb)
            Y_list.append(gt_rates)

            Xdepthb = np.stack([sw_depths, vid_depths], axis=1)
            X_depth_list.append(Xdepthb)
            Y_depth_list.append(gt_depths)

            print("--" * 20)
            print("**" * 20)


        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)

        # fit linear fusion (no intercept)
        fusion = LinearRegression(fit_intercept=False)
        fusion.fit(X, Y)
        w_sw, w_vid = fusion.coef_
        print(f"Learned weights -> smartwatch: {w_sw:.3f}, video: {w_vid:.3f}")

        # save weights
        np.savetxt(f"./weights/{job}_fusion_weights.txt", fusion.coef_)
        print("Fusion weights saved to fusion_weights.txt")

        # do the linear fusion on the depth data
        X_depth = np.vstack(X_depth_list)
        Y_depth = np.concatenate(Y_depth_list)

        fusion_depth = LinearRegression(fit_intercept=False)
        fusion_depth.fit(X_depth, Y_depth)
        w_sw_depth, w_vid_depth = fusion_depth.coef_
        print(f"Learned depth weights -> smartwatch: {w_sw_depth:.3f}, video: {w_vid_depth:.3f}")
        # save depth weights
        np.savetxt(f"./weights/{job}_fusion_depth_weights.txt", fusion_depth.coef_)
        print("Fusion depth weights saved to fusion_depth_weights.txt")

        # save the smartwatch model
        torch.save(smartwatch_model.state_dict(), f"./weights/{job}_smartwatch_model.pth")
        print(f"Smartwatch model saved to {job}_smartwatch_model.pth")



if __name__ == "__main__":
    main()
