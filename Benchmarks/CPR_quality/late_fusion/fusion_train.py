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
from ego_rate_estimator.cpr_rate_detection_video import ego_rate_detect, init_models
from smartwatch_rate_estimator.cpr_rate_detection_smartwatch import smartwatch_rate_detect, get_gt_cpr_rate

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import argparse
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_rate(frames, smartwatch, gt, window_size, video_id):
    """
    Slide a tumbling window and compute:
      - smartwatch_rate per window
      - ego_video_rate per window
      - gt_rate per window
    Returns three 1D numpy arrays: (sw_rates, vid_rates, gt_rates)
    """
    sw_rates, vid_rates, gt_rates = [], [], []
    total = frames.shape[0]

    for start in range(0, total, window_size):
        end = start + window_size
        if end > total:
            break

        try:
                
            win_frames = frames[start:end]
            win_smart = smartwatch[start:end]
            win_gt = gt[start:end]

            v_rate = ego_rate_detect(win_frames, video_id)
            s_rate = smartwatch_rate_detect(win_smart)
            g_rate = get_gt_cpr_rate(win_gt)


            # ensure Python floats
            try:
                v_rate = float(v_rate)
                s_rate = float(s_rate)
                g_rate = float(g_rate)
            except (TypeError, ValueError):
                continue

            # skip NaNs
            if np.isnan(v_rate) or np.isnan(s_rate) or np.isnan(g_rate):
                continue

            sw_rates.append(s_rate)
            vid_rates.append(v_rate)
            gt_rates.append(g_rate)

            print(f"window:{start}:{end},sw_rate:{s_rate:.2f},ego_rate:{v_rate:.2f},gt_rate:{g_rate:.2f}")

        except Exception as e:
            print(f"Error processing window {start}:{end} for video {video_id}: {e}")
            continue
    return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)

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

    # create a CSV file path
    csv_path = f"./results/{job}_train_fusion_rates_results.csv"

    # remove if already exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # accumulate per-window rates across all subjects
    X_list, Y_list = [], []
    for batch in train_loader:

        print("Subject: ", batch['subject_id'])
        print("Trial", batch['trial_id'])
        input, feature_size, gt_sensor_data = preprocess(batch, modality, None, device, task=task)

        video_id = f"{batch['subject_id']}_{batch['trial_id']}_ego"

        frames = input['frames'][0]    # [T,3,H,W]
        smartwatch = input['smartwatch'][0]    # [T,3]
        gt_sensor_data = gt_sensor_data[0]     # [T,3]

        sw_rates, vid_rates, gt_rates = detect_rate(frames, smartwatch, gt_sensor_data, window, video_id)

        # average the rates over the window
        s_rate = np.mean(sw_rates)
        v_rate = np.mean(vid_rates)
        g_rate = np.mean(gt_rates)

        # Create a row and save to CSV immediately
        row = pd.DataFrame([{
            "subject_id": batch['subject_id'],
            "trial_id": batch['trial_id'],
            "smartwatch_rate": s_rate,
            "video_rate": v_rate,
            "gt_rate": g_rate
        }])

        # Write to CSV, with header only on first write
        write_header = not os.path.exists(csv_path)
        row.to_csv(csv_path, mode='a', index=False, header=write_header)

        Xb = np.stack([sw_rates, vid_rates], axis=1)
        X_list.append(Xb)
        Y_list.append(gt_rates)

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


if __name__ == "__main__":
    main()
