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

from datetime import datetime

from scripts.config import DefaultArgsNamespace
from datautils.ems import *
from ego_rate_estimator.cpr_rate_detection_video import init_models, ego_rate_detect,ego_rate_detect_cached
from smartwatch_rate_estimator.cpr_rate_detection_smartwatch import smartwatch_rate_detect, get_gt_cpr_rate
from utils.utils import preprocess, detect_rate


from ego_depth_estimator.supervised_depth_estimate_window_ego import initialize_midas_model
from smartwatch_depth_estimator.cpr_depth_detection_smartwatch import initialize_smartwatch_model



# cache dir for ego video hand keypoints
CACHE_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/GoPro_CPR_Clips/ego_gopro_cpr_clips/test_root/chest_compressions/"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def detect_rate(frames, smartwatch, gt, window_size, video_id):
#     """Compute per-window rates for one clip."""
#     sw_rates, vid_rates, gt_rates = [], [], []
#     total = frames.shape[0]
#     for start in range(0, total, window_size):
#         end = start + window_size
#         if end > total:
#             break
#         win_frames = frames[start:end]
#         win_sw = smartwatch[start:end]
#         win_gt = gt[start:end]

#         # estimates
#         # v = ego_rate_detect(win_frames, video_id)
#         v = ego_rate_detect_cached(win_frames, video_id, start, end, cache_dir=CACHE_DIR)
#         s = smartwatch_rate_detect(win_sw)
#         g = get_gt_cpr_rate(win_gt)

#         print(f"Window {start}-{end}: Video rate: {v}, Smartwatch rate: {s}, GT rate: {g}")

#         try:
#             v = float(v); s = float(s); g = float(g)
#         except:
#             continue
#         if np.isnan(v) or np.isnan(s) or np.isnan(g):
#             continue

#         sw_rates.append(s)
#         vid_rates.append(v)
#         gt_rates.append(g)
#     return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rate_weights_file', required=True)
    parser.add_argument('--depth_weights_file', required=True)
    parser.add_argument('--sw_model_checkpoint', required=True)
    parser.add_argument('--job_id', type=str, default="default")
    args = parser.parse_args()
    job = args.job_id
    print(f"Job ID: {job}")

    args = parser.parse_args()

    # initialize models
    init_models()

    # load fusion weights for cpr rate estimation
    w_rate_sw, w_rate_vid = np.loadtxt(args.rate_weights_file)

    # load fusion weights for depth estimation
    w_depth_sw, w_depth_vid = np.loadtxt(args.depth_weights_file)

    job_id_from_weights = os.path.basename(args.sw_model_checkpoint).split(".")[0]

    # get dataloader
    ns = DefaultArgsNamespace()
    _, _, test_loader, _, _ = eee_get_dataloaders(ns)
    window = args.window_size
    modality = ns.dataloader_params['modality']
    task = ns.dataloader_params['task']

    print(f"Using weights: Smartwatch: {w_rate_sw}, Video: {w_rate_vid}")

    # get current date and time
    # date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create results directory if it doesn't exist
    results_dir = f"./results/fusion_test_{job}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    else:
        print(f"Results directory already exists: {results_dir}")

    # prepare CSV
    out_csv = f"{results_dir}/test_results_{job_id_from_weights}_test_fusion_rmse_per_batch.csv"
    if os.path.exists(out_csv):
        os.remove(out_csv)
    df = pd.DataFrame(columns=["subject_id","trial_id","smartwatch_rate","video_rate","fused_rate","gt_rate","rmse_cpm", "smartwatch_depth", "video_depth", "fused_depth", "gt_depth", "rmse_depth"])
    df.to_csv(out_csv, index=False)

    debug_csv = f"{results_dir}/debug_{job_id_from_weights}_test_fusion_debug_per_batch.csv"
    debug_df = pd.DataFrame(columns=["subject_id", "trial_id", "start_idx", "smartwatch_rates", "video_rates", "gt_rates", "fused_rates", "smartwatch_depths", "video_depths", "gt_depths", "fused_depths"])
    debug_df.to_csv(debug_csv, mode='a', header=False, index=False)

    print(f"Created output CSV: {out_csv}")

    # initialize models

    # define variable for model_size
    model_type="DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type="DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type="MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # initializing midas model for depth estimation
    midas_model, midas_transform, SCALE_FACTOR = initialize_midas_model(model_type=model_type, device=device)

    # initialize smartwatch models
    smartwatch_model, smartwatch_optimizer, smartwatch_criterion, smartwatch_scheduler = initialize_smartwatch_model(device=device)

    # load smartwatch model weights
    smartwatch_model.load_state_dict(torch.load(args.sw_model_checkpoint, map_location=device))
    smartwatch_model.eval()


    # iterate batches
    for batch in test_loader:
        print("-" * 50)

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

        inp, _, gt_data = preprocess(batch, modality, None, device=torch.device("cpu"), task=task)
        frames = inp['frames'][0]
        if torch.is_tensor(frames):
            frames = frames.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        sw = inp['smartwatch'][0]
        if torch.is_tensor(sw): sw = sw.cpu().numpy()
        gt = gt_data[0]
        if torch.is_tensor(gt): gt = gt.cpu().numpy()

        sw_rates, vid_rates, gt_rates = detect_rate(frames, sw, gt, window, video_id, CACHE_DIR)

        print(f"Smartwatch rates: {sw_rates}")
        print(f"Video rates: {vid_rates}")
        print(f"GT rates: {gt_rates}")

        smartwatch = inp['smartwatch'][0]    # [T,3]
        gt_sensor_data = gt_data[0]     # [T,3]

        sw_depths, vid_depths, gt_depths = detect_depth(frames, smartwatch, gt_sensor_data, window, video_id, CACHE_DIR, midas_model, midas_transform, device, SCALE_FACTOR,  smartwatch_model, smartwatch_optimizer, smartwatch_criterion, MODE="test")

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

  
        # fuse and compute RMSE
        fused = w_rate_sw * sw_rates + w_rate_vid * vid_rates

        # fuse depths
        fused_depths = w_depth_sw * sw_depths + w_depth_vid * vid_depths

        # write the rates to CSV
        debug_row = {
            "subject_id": subj,
            "trial_id": trial,
            "start_idx": start_idx,
            "smartwatch_rates": sw_rates.tolist(),
            "video_rates": vid_rates.tolist(),
            "gt_rates": gt_rates.tolist(),
            "fused_rates": fused.tolist(),
            "smartwatch_depths": sw_depths.tolist(),
            "video_depths": vid_depths.tolist(),
            "gt_depths": gt_depths.tolist(),
            "fused_depths": fused_depths.tolist()
        }

        debug_df = pd.DataFrame([debug_row])
        debug_df.to_csv(debug_csv, mode='a', header=False, index=False)

        rmse = np.sqrt(mean_squared_error(gt_rates, fused))

        rmse_depth = np.sqrt(mean_squared_error(gt_depths, fused_depths))

        # append to CSV
        row = {"subject_id": subj, "trial_id": trial, "smartwatch_rate":np.mean(sw_rates), "video_rate":np.mean(vid_rates),"fused_rate":np.mean(fused) , "gt_rate":np.mean(gt_rates) ,"rmse_cpm": float(rmse), "smartwatch_depth": np.mean(sw_depths), "video_depth": np.mean(vid_depths), "fused_depth": np.mean(fused_depths), "gt_depth": np.mean(gt_depths), "rmse_depth": float(rmse_depth)}


        pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
        # print all rate estimates for subject/trial
        print(f"Subject: {subj}, Trial: {trial}")
        print(f"Smartwatch rates: {sw_rates}")
        print(f"Video rates: {vid_rates}")
        print(f"GT rates: {gt_rates}")
        print(f"Fused rates: {fused}")
        print(f"GT RMSE: {rmse:.2f} CPM")
        print(f"Smartwatch depths: {sw_depths}")
        print(f"Video depths: {vid_depths}")
        print(f"GT depths: {gt_depths}")
        print(f"Fused depths: {fused_depths}")
        print(f"GT Depth RMSE: {rmse_depth:.2f} mm")
        print("-" * 50)

    print(f"Results saved to {out_csv}")

if __name__ == "__main__":
    main()