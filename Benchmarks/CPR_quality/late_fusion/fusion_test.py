#!/usr/bin/env python3
"""
Test a trained fusion model for CPR rate estimation:
Compute and save RMSE per batch (subject/trial).

Usage:
    python fusion_test_per_batch.py \
        --annotation_file Annotations/main_annotation_cpr_quality.json \
        --data_base_path /path/to/ego_exo_data \
        --split split_1 \
        --window_size 150 \
        --batch_size 1 \
        --weights_file fusion_weights.txt
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from utils.utils import *

from scripts.config import DefaultArgsNamespace
from datautils.ems import *
from ego_rate_estimator.cpr_rate_detection_video import init_models, ego_rate_detect
from smartwatch_rate_estimator.cpr_rate_detection_smartwatch import smartwatch_rate_detect, get_gt_cpr_rate
from utils.utils import preprocess

def detect_rate(frames, smartwatch, gt, window_size, video_id):
    """Compute per-window rates for one clip."""
    sw_rates, vid_rates, gt_rates = [], [], []
    total = frames.shape[0]
    for start in range(0, total, window_size):
        end = start + window_size
        if end > total:
            break
        win_frames = frames[start:end]
        win_sw = smartwatch[start:end]
        win_gt = gt[start:end]

        # estimates
        v = ego_rate_detect(win_frames, video_id)
        s = smartwatch_rate_detect(win_sw)
        g = get_gt_cpr_rate(win_gt)

        try:
            v = float(v); s = float(s); g = float(g)
        except:
            continue
        if np.isnan(v) or np.isnan(s) or np.isnan(g):
            continue

        sw_rates.append(s)
        vid_rates.append(v)
        gt_rates.append(g)
    return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weights_file', required=True)
    parser.add_argument('--job_id', type=str, default="default")
    args = parser.parse_args()
    job = args.job_id
    print(f"Job ID: {job}")

    args = parser.parse_args()

    # initialize models
    init_models()

    # load fusion weights
    w_sw, w_vid = np.loadtxt(args.weights_file)

    job_id_from_weights = os.path.basename(args.weights_file).split(".")[0]

    # get dataloader
    ns = DefaultArgsNamespace()
    _, _, test_loader, _, _ = eee_get_dataloaders(ns)
    window = args.window_size
    modality = ns.dataloader_params['modality']
    task = ns.dataloader_params['task']

    # prepare CSV
    out_csv = f"./results/{job_id_from_weights}_test_fusion_rmse_per_batch.csv"
    if os.path.exists(out_csv):
        os.remove(out_csv)
    df = pd.DataFrame(columns=["subject_id","trial_id","smartwatch_rate","video_rate","gt_rate","rmse_cpm"])
    df.to_csv(out_csv, index=False)

    print(f"Created output CSV: {out_csv}")

    # iterate batches
    for batch in test_loader:
        print("-" * 50)

        subj = batch['subject_id']
        trial = batch['trial_id']
        video_id = f"{subj}_{trial}_ego"

        inp, _, gt_data = preprocess(batch, modality, None, device=torch.device("cpu"), task=task)
        frames = inp['frames'][0]
        if torch.is_tensor(frames):
            frames = frames.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        sw = inp['smartwatch'][0]
        if torch.is_tensor(sw): sw = sw.cpu().numpy()
        gt = gt_data[0]
        if torch.is_tensor(gt): gt = gt.cpu().numpy()

        sw_rates, vid_rates, gt_rates = detect_rate(frames, sw, gt, window, video_id)
        if sw_rates.size == 0:
            print(f"{subj}_{trial}: no valid windows, skipping.")
            continue

        # fuse and compute RMSE
        fused = w_sw * sw_rates + w_vid * vid_rates
        rmse = np.sqrt(mean_squared_error(gt_rates, fused))

        # append to CSV
        row = {"subject_id": subj, "trial_id": trial, "smartwatch_rate":np.mean(sw_rates), "video_rate":np.mean(vid_rates), "gt_rate":np.mean(gt_rates) ,"rmse_cpm": float(rmse)}
        pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
        # print all rate estimates for subject/trial
        print(f"Subject: {subj}, Trial: {trial}")
        print(f"Smartwatch rates: {sw_rates}")
        print(f"Video rates: {vid_rates}")
        print(f"GT rates: {gt_rates}")
        print(f"Fused rates: {fused}")
        print(f"GT RMSE: {rmse:.2f} CPM")
        print("-" * 50)

    print(f"Results saved to {out_csv}")

if __name__ == "__main__":
    main()
