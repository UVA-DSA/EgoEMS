import os
import sys
import torch
import math
import numpy as np
from torch.utils.data import DataLoader

import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)

from scipy.signal import find_peaks, lombscargle
from EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import utils
from tools import tools as depth_tools
import matplotlib.pyplot as plt

DATA_FPS = 30
bs = 1
CLIP_LENGTH = 5

MIN_ACC = torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0)
MAX_ACC = torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0)
MIN_DEPTH = 0.0
MAX_DEPTH = 110.0

debug_output_folder = r'Benchmarks/CPR_quality/smartwatch/validation_plots/'
os.makedirs(debug_output_folder, exist_ok=True)

annot_path = r'Annotations/main_annotation_cpr_quality.json'
split_paths = [
    r'Annotations/splits/cpr_quality/split_1.json', 
    r'Annotations/splits/cpr_quality/split_2.json', 
    r'Annotations/splits/cpr_quality/split_3.json', 
    r'Annotations/splits/cpr_quality/split_4.json'
]

log_base_path = r'Benchmarks/CPR_quality/smartwatch/logs/'
model_save_base_path = r'Benchmarks/CPR_quality/smartwatch/checkpoints/'
model_save_path = r''
data_path = r''

def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg + '\n')

def get_avg_depth(depth_gt):
    avg_depth_list, min_depth_list, n_cpr = [], [], []
    if len(depth_gt.shape) == 1:
        depth_gt = depth_gt.unsqueeze(0)
    for i in range(depth_gt.shape[0]):
        p, v = depth_tools.detect_peaks_and_valleys_depth_sensor(depth_gt[i, :].detach().numpy(), show=False)
        peak_depths = depth_gt[i, p]
        valley_depths = depth_gt[i, v]
        # how many complete peak–valley pairs?
        l = min(len(peak_depths), len(valley_depths))
        if l == 0:
            # no valid CPR cycles in this segment
            avg_depth_list.append(0.0)
            min_depth_list.append(0.0)
            n_cpr.append(0.0)
        else:
            diffs = peak_depths[:l] - valley_depths[:l]
            avg_depth_list.append(diffs.mean().item())
            min_depth_list.append(valley_depths[:l].min().item())
            n_cpr.append((len(p) + len(v)) * 0.5)
    return torch.tensor(avg_depth_list), torch.tensor(min_depth_list), torch.tensor(n_cpr)

def calculate_cpr_rate_with_lombscargle(accel_magnitude, timestamps, window, stride):
    steps = math.floor((len(accel_magnitude) - window) / stride)
    cprRates = []
    print("Steps:", steps)
    for step in range(steps):

        start = step * stride
        stop = start + window
        x = accel_magnitude[start:stop]
        t = timestamps[start:stop]
        
        n = len(t)
        duration = t.ptp()
        freqs = np.linspace(1 / duration, n / duration, 5 * n)
        periodogram = lombscargle(t, x, freqs)
        mags = periodogram
        freqs = freqs / (2 * math.pi)

        # Find peak frequency excluding DC component at 0 Hz
        peak_point = np.argmax(mags[1:]) + 1
        cpr_rate = freqs[peak_point] * 60
        cprRates.append(cpr_rate)

    average_cpr_rate = np.mean(cprRates)
    return average_cpr_rate

def plot_and_save(data, depth_gt, timestamps, gt_cpr_rate, cpr_rate, batch_idx, subject, trial, suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, depth_gt, label="Depth Sensor Data", color="blue")
    plt.plot(timestamps, data, label="Smartwatch Accel Magnitude", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title(f"GT Rate: {gt_cpr_rate:.2f} CPM vs Calculated Rate: {cpr_rate:.2f} CPM")
    plt.legend()
    
    # Save the plot
    output_path = os.path.join(debug_output_folder, f"{suffix}batch_{batch_idx}_subject_{subject}_trial_{trial}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def normalize_depth(depth_gt, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
    return (depth_gt - min_depth) / (max_depth - min_depth)


def generate_cpr_rate(train_data_loader, valid_data_loader, log_path):
    for i, batch in enumerate(valid_data_loader):
        print("--------------------")
        print(f"Processing validation batch {i}/{len(valid_data_loader)}")
        data = batch['smartwatch'].float()
        depth_gt = batch['depth_sensor'].squeeze()

    
        # metadata 
        subject = batch['subject_id']
        trial = batch['trial_id']
        
        # Calculate CPR rate with Lomb-Scargle
        window = DATA_FPS 
        stride = 5

        accel_magnitude = torch.sqrt(torch.sum(data ** 2, axis=2))
        accel_magnitude_np = accel_magnitude.numpy().flatten()
        timestamps = np.linspace(0, len(accel_magnitude_np) / DATA_FPS, len(accel_magnitude_np))
        print("Accelerometer magnitude shape:", accel_magnitude.shape)
        print("Timestamps shape:", timestamps.shape)

        low_pass_accel_magnitude = depth_tools.low_pass_filter(accel_magnitude)
        low_pass_accel_magnitude_np = low_pass_accel_magnitude.numpy().flatten()

        # apply low pass filter
        

        # avg_depths_list, min_depths_list, gt_cpr_rate = get_avg_depth(depth_gt)
        # calculate ground truth cpr rate with lombscargle 
        # gt_cpr_rate = calculate_cpr_rate_with_lombscargle(depth_gt, timestamps, window, stride)
        # print("GT CPR rate:", gt_cpr_rate)

        #  method
        avg_depths_list, min_depths_list, keshara_gt_cpr_rate = get_avg_depth(depth_gt)
        print("GT CPR rate:", keshara_gt_cpr_rate)

        # cpr_rates = calculate_cpr_rate_with_lombscargle(accel_magnitude_np, timestamps, window, stride)
        # print("Calculated CPR rates:", cpr_rates)

        # no filtering
        avg_depths_list, min_depths_list, keshara_pred_cpr_rate = get_avg_depth(accel_magnitude)
        print("Pred CPR rate:", keshara_pred_cpr_rate)

        # low pass filtering
        avg_depths_list, min_depths_list, low_pass_keshara_pred_cpr_rate = get_avg_depth(low_pass_accel_magnitude)
        print("Low Pass Pred CPR rate:", low_pass_keshara_pred_cpr_rate)


        log_msg = f"Subject: {subject}, Trial: {trial}, GT CPR Rate: {keshara_gt_cpr_rate.tolist()}, Calculated CPR Rates: {keshara_pred_cpr_rate.tolist()}"
        write_log_line(log_path,log_msg)

        # plot and save
        plot_and_save(accel_magnitude_np, depth_gt, timestamps, keshara_gt_cpr_rate.item(), keshara_pred_cpr_rate.item(), i, subject, trial)
        plot_and_save(low_pass_accel_magnitude_np, depth_gt, timestamps, keshara_gt_cpr_rate.item(), low_pass_keshara_pred_cpr_rate.item(), i, subject, trial, "low_pass_")
        print("--------------------")


if __name__ == "__main__":
    split_path = split_paths[3]
    split = split_path.split('/')[-1].split('.')[0]
    log_path = os.path.join(log_base_path, f'cpr_rate_detection_log_{split}.txt')
    
    init_log(log_path)
    
    data = EgoExoEMSDataset(
        annotation_file=annot_path,
        data_base_path="",
        fps=DATA_FPS,
        frames_per_clip=DATA_FPS * CLIP_LENGTH,
        data_types=['smartwatch', 'depth_sensor'],
        split='train',
        activity='chest_compressions',
        split_path=split_path
    )
    train_data_loader = DataLoader(data, batch_size=bs, shuffle=False)
    print("Train data loader length:", len(train_data_loader))
    
    data = EgoExoEMSDataset(
        annotation_file=annot_path,
        data_base_path=data_path,
        fps=DATA_FPS,
        frames_per_clip=DATA_FPS * CLIP_LENGTH,
        data_types=['smartwatch', 'depth_sensor'],
        split='validation',
        activity='chest_compressions',
        split_path=split_path
    )
    valid_data_loader = DataLoader(data, batch_size=bs, shuffle=False)
    print("Validation data loader length:", len(valid_data_loader))
    
    generate_cpr_rate(train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, log_path=log_path)


# a function to detect the cpr rate from smartwatch data given a window of data
# and return the cpr rate
def smartwatch_rate_detect(smartwatch_window):
    """
    Estimate CPR rate (compressions per minute) from one window of raw
    3‑axis accelerometer data.

    Args:
        smartwatch_window (np.ndarray or torch.Tensor): shape (N,3) or (1,N,3)
    Returns:
        float: estimated compressions per minute (CPM)
    """
    # 1) Turn input into a (1, N, 3) torch tensor
    if not torch.is_tensor(smartwatch_window):
        data = torch.tensor(smartwatch_window, dtype=torch.float32)
    else:
        data = smartwatch_window.float()
    if data.ndim == 2:             # (N,3) → (1,N,3)
        data = data.unsqueeze(0)
    elif data.ndim == 3 and data.shape[0] != 1:
        raise ValueError("Expected window shape (N,3) or (1,N,3), got %s" % (data.shape,))

    # 2) Compute per‑sample magnitude → shape (1, N)
    accel_mag = torch.norm(data, dim=2)

    # 3) Apply low-pass filter
    low_pass_accel_magnitude = depth_tools.low_pass_filter(accel_mag)


    # 3) Detect peaks/valleys and get raw # of compressions
    #    get_avg_depth returns (avg_depths, min_depths, n_cpr)
    _, _, n_cpr = get_avg_depth(low_pass_accel_magnitude)

    # avg_depths_list, min_depths_list, low_pass_keshara_pred_cpr_rate = get_avg_depth(low_pass_accel_magnitude)
    # print("Low Pass Pred CPR rate:", low_pass_keshara_pred_cpr_rate)

    # 4) Convert count → rate (CPM)
    window_secs = accel_mag.shape[1] / DATA_FPS
    cpr_rate = n_cpr.item() * (60.0 / window_secs)

    return cpr_rate

def get_gt_cpr_rate(gt_window):
    # 1) Turn input into a (1, N, 3) torch tensor

    print("GT Window shape:", gt_window.shape)
    if not torch.is_tensor(gt_window):
        data = torch.tensor(gt_window, dtype=torch.float32)
    else:
        data = gt_window.float()
    if data.ndim == 2:             # (N,3) → (1,N,3)
        data = data.unsqueeze(0)
    elif data.ndim == 3 and data.shape[0] != 1:
        raise ValueError("Expected window shape (N,3) or (1,N,3), got %s" % (data.shape,))

    # 2) Compute per‑sample magnitude → shape (1, N)
    accel_mag = torch.norm(data, dim=2)


    # 3) Detect peaks/valleys and get raw # of compressions
    #    get_avg_depth returns (avg_depths, min_depths, n_cpr)
    _, _, n_cpr = get_avg_depth(accel_mag)

    # 4) Convert count → rate (CPM)
    window_secs = accel_mag.shape[1] / DATA_FPS
    cpr_rate = n_cpr.item() * (60.0 / window_secs)

    return cpr_rate


def get_gt_cpr_depth(gt_window):
    # 1) Turn input into a (1, N, 3) torch tensor

    print("GT Window shape:", gt_window.shape)
    if not torch.is_tensor(gt_window):
        data = torch.tensor(gt_window, dtype=torch.float32)
    else:
        data = gt_window.float()
    if data.ndim == 2:             # (N,3) → (1,N,3)
        data = data.unsqueeze(0)
    elif data.ndim == 3 and data.shape[0] != 1:
        raise ValueError("Expected window shape (N,3) or (1,N,3), got %s" % (data.shape,))

    # 2) Compute per‑sample magnitude → shape (1, N)
    accel_mag = torch.norm(data, dim=2)


    # gt_peaks, gt_valleys = depth_tools.detect_peaks_and_valleys_depth_sensor(gt_window, mul=1, show=False)
    # n_cpr_window_gt = (len(gt_peaks) + len(gt_valleys)) * 0.5

    # peak_depths_gt = gt_window[gt_peaks]
    # valley_depths_gt = gt_window[gt_valleys]
    # l = min(len(peak_depths_gt), len(valley_depths_gt))
    # gt_cpr_depth = float((peak_depths_gt[:l] - valley_depths_gt[:l]).mean()) if l > 0 else 0

    # print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")


    #    get_avg_depth returns (avg_depths, min_depths, n_cpr)
    avg_depths, _, n_cpr = get_avg_depth(accel_mag)

    gt_cpr_depth = avg_depths.mean().item()

    print(f"GT CPR Depth (mm): {gt_cpr_depth:.2f}")


    return  avg_depths, _, gt_cpr_depth