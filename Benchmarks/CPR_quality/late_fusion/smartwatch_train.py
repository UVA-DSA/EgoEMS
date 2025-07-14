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


from utils.utils import preprocess
from smartwatch_depth_estimator.cpr_depth_detection_smartwatch import initialize_smartwatch_model, initialize_new_smartwatch_model

MIN_DEPTH=0.0
MAX_DEPTH=120.0


def train_smartwatch_depth_estimator(model, train_loader, optimizer, criterion, device, modality):
    task = "cpr_quality"
    model.train()

    # Define constants once
    MIN_ACC = torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0).to(device)
    MAX_ACC = torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0).to(device)

    

    for batch in train_loader:
        print("**" * 20)
        print("--" * 20)

        subj = batch['subject_id'][0]
        trial = batch['trial_id'][0]
        start_t = batch['start_t'][0]
        end_t = batch['end_t'][0]
        keystep_id = batch['keystep_id'][0]

        print(f"Processing Subject: {subj}, Trial: {trial}, Keystep_ID: {keystep_id}, Start: {start_t}, End: {end_t}")

        video_id = f"{subj}_t{trial}_ks{keystep_id}_{float(start_t):.3f}_{float(end_t):.3f}"
        print(f"Video ID: {video_id}")

        input, feature_size, gt_sensor_data = preprocess(batch, modality, None, device, task=task)


        print("Input shape:", input.shape, "gt_sensor_data shape:", gt_sensor_data.shape)
        total = input.shape[1]
        input = input[0]

        gt_sensor_data = gt_sensor_data[0]

        print("\n\n" + "=" * 50)
        print(f"[TRAINING] Processing video {video_id} with {total} samples")

        window_size = 150

        for start in range(0, total, window_size):
            end = start + window_size
            if end > total:
                break

            print("-*" * 20)
            print(f"[CPR DEPTH DETECTION]:Processing window {start}:{end} for video {video_id}...")

            try:
                win_smart = input[start:end]
                win_gt = gt_sensor_data[start:end]

                # Check preprocess output

                window_smartwatch = win_smart
                window_gt_sensor_data = win_gt
                
                if torch.isnan(window_smartwatch).any() or torch.isinf(window_smartwatch).any():
                    print("⚠️ NaN or Inf in raw smartwatch data! Skipping batch.")
                    continue


                avg_depths_list, min_depths_list, gt_cpr_depth = get_gt_cpr_depth(window_gt_sensor_data.cpu().numpy())
                comp_depth = avg_depths_list / MAX_DEPTH

                        #normnalize depth
                depth_gt_norm=(window_gt_sensor_data.cpu()-min_depths_list.unsqueeze(1))/MAX_DEPTH

                
                window_smartwatch = window_smartwatch.to(device)
                depth_gt_norm = depth_gt_norm.to(device)
                comp_depth = comp_depth.to(device)

                imu = window_smartwatch              # [T=150, C=3]
                imu_min, _ = imu.min(dim=0, keepdim=True)  # [1,3]
                imu_max, _ = imu.max(dim=0, keepdim=True)
                data_norm = (imu - imu_min) / (imu_max - imu_min + 1e-6)
                data_norm = data_norm.unsqueeze(0)

                # data_norm = (window_smartwatch - MIN_ACC) / (MAX_ACC - MIN_ACC)

                rec, depth_pred = model(data_norm.permute(0, 2, 1))
                print("Smartwatch depth pred mean:", depth_pred.mean().item())
                print("GT depth norm mean:", depth_gt_norm.mean().item())


                # Unsqueeze scalar to shape [1]
                if depth_pred.dim() == 0:
                    depth_pred = depth_pred.unsqueeze(0)

                if comp_depth.dim() == 0:
                    comp_depth = comp_depth.unsqueeze(0)

                d_loss = criterion(depth_pred, comp_depth)


                loss =  d_loss
                print("Total loss:", loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                depth_pred = depth_pred * MAX_DEPTH  # scale back

                print(f"Final Depth Prediction: {depth_pred.mean().item()}")
            
            except Exception as e:
                print(f"Error processing window {start}:{end} for video {video_id}: {e}")
                continue


def test_smartwatch_depth_estimator(model, test_loader, criterion, device, modality, results_csv_path="./test_predictions.csv"):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    MIN_ACC = torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0).to(device)
    MAX_ACC = torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0).to(device)

    results_rows = []

    with torch.no_grad():
        for batch in test_loader:

            input, feature_size, gt_sensor_data = preprocess(batch, modality, None, device, task="cpr_quality")


            subj = batch['subject_id'][0]
            trial = batch['trial_id'][0]
            start_t = batch['start_t'][0]
            end_t = batch['end_t'][0]
            keystep_id = batch['keystep_id'][0]
            video_id = f"{subj}_t{trial}_ks{keystep_id}_{float(start_t):.3f}_{float(end_t):.3f}"

            print("--" * 20)
            print("[TEST] Processing Subject: {}, Trial: {}, Keystep_ID: {}, Start: {}, End: {}".format(
                subj, trial, keystep_id, start_t.item(), end_t.item()))
            
            print("Input shape:", input.shape)
            total = input.shape[1]

            input = input[0]


            gt_sensor_data = gt_sensor_data[0]

            print("\n\n" + "=" * 50)
            print(f"Processing video {video_id} with {total} samples.")

            window_size = 150

            for start in range(0, total, window_size):
                end = start + window_size
                if end > total:
                    break

                try:

                    win_smart = input[start:end]
                    win_gt = gt_sensor_data[start:end]

                    # Check preprocess output

                    window_smartwatch = win_smart
                    window_gt_sensor_data = win_gt
                    

                    if torch.isnan(window_smartwatch).any() or torch.isinf(window_smartwatch).any():
                        print("⚠️ NaN or Inf in raw smartwatch data! Skipping batch.")
                        continue

    
                    avg_depths_list, min_depths_list, gt_cpr_depth = get_gt_cpr_depth(window_gt_sensor_data.cpu().numpy())
                    comp_depth = avg_depths_list / MAX_DEPTH



                    depth_gt_norm = (window_gt_sensor_data.cpu() - min_depths_list.unsqueeze(1)) / MAX_DEPTH

                    window_smartwatch = window_smartwatch.to(device)
                    depth_gt_norm = depth_gt_norm.to(device)
                    comp_depth = comp_depth.to(device)

                    imu = window_smartwatch              # [T=150, C=3]
                    imu_min, _ = imu.min(dim=0, keepdim=True)  # [1,3]
                    imu_max, _ = imu.max(dim=0, keepdim=True)
                    data_norm = (imu - imu_min) / (imu_max - imu_min + 1e-6)
                    data_norm = data_norm.unsqueeze(0)
                    # data_norm = (smartwatch - MIN_ACC) / (MAX_ACC - MIN_ACC + 1e-6)

                    rec, depth_pred = model(data_norm.permute(0, 2, 1))

                    depth_gt_mask = gt_sensor_data > 0


                    if depth_pred.dim() == 0:
                        depth_pred = depth_pred.unsqueeze(0)

                    if comp_depth.dim() == 0:
                        comp_depth = comp_depth.unsqueeze(0)
                        
                    d_loss = criterion(depth_pred, comp_depth)

                    loss =  d_loss
                    total_loss += loss.item() * len(window_smartwatch)
                    total_samples += len(window_smartwatch)

                    depth_pred_value = (depth_pred * MAX_DEPTH).cpu().item()
                    gt_cpr_depth_value = gt_cpr_depth if isinstance(gt_cpr_depth, float) else gt_cpr_depth.item()
                    
                    # Record one row
                    results_rows.append({
                        "subject_id": subj,
                        "trial_id": trial,
                        "start_t": start_t.item(),
                        "end_t": end_t.item(),
                        "keystep_id": keystep_id.item(),
                        "video_id": video_id,
                        "gt_cpr_depth": gt_cpr_depth_value,
                        "depth_pred": depth_pred_value
                    })

                    print(f"Final Depth Prediction: {depth_pred_value}, GT Depth: {gt_cpr_depth_value}")

                except Exception as e:
                    print(f"Error processing window {start}:{end} for video {video_id}: {e}")
                    continue

    # Save results to CSV
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Saved predictions CSV to {results_csv_path}")

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    print(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss





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

    # overload modality to only use smartwatch and depth sensor
    modality = ['smartwatch', 'depth_sensor']
    args.dataloader_params['modality'] = modality
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
    debug_path = f"./debug/smartwatch_train_{date_time_str}"

    # create results directory if it doesn't exist
    results_dir = f"./results/smartwatch_train_{job}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    else:
        print(f"Results directory already exists: {results_dir}")


    # initialize smartwatch models
    smartwatch_model, smartwatch_optimizer, smartwatch_criterion, smartwatch_scheduler = initialize_smartwatch_model(device=device)


    debug_csv = f"{results_dir}/debug_{job}_train_smartwatch_debug_per_batch.csv"
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

    best_test_loss = float('inf')
    # loop for few epochs
    print("Starting training...")
    for epoch in range(10):  # just one epoch for now

        # accumulate per-window rates across all subjects
        X_list, Y_list = [], []
        X_depth_list, Y_depth_list = [], []

        print(f"Epoch {epoch + 1}/{10}")

        # call train function
        train_smartwatch_depth_estimator(smartwatch_model, train_loader, smartwatch_optimizer, smartwatch_criterion, device, modality)

        # smoke test the model

        # test the model
        test_loss = test_smartwatch_depth_estimator(smartwatch_model, test_loader, smartwatch_criterion, device, modality, results_csv_path=f"{results_dir}/test_predictions_@_epoch_{epoch + 1}_{job}.csv")
        print(f"Test Loss after epoch {epoch + 1}: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f"New best test loss: {best_test_loss:.4f}")
            # Save the model
            torch.save(smartwatch_model.state_dict(), f"./weights/best_smartwatch_model_{job}.pth")
            print(f"Saved best model to weights/best_smartwatch_model_{job}.pth")


if __name__ == "__main__":
    print("Starting smartwatch training script...")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print("Smartwatch training script executed.")