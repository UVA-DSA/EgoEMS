import torch
import os
import sys
import math
import numpy as np
from torch.utils.data import DataLoader


import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)


from tools import tools as depth_tools

from smartwatch_depth_estimator.SWnet import SWNET, TCNDepthNet

MIN_ACC=torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0)
MAX_ACC=torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0)

MIN_DEPTH=0.0
MAX_DEPTH=82.0

DATA_FPS=30
CLIP_LENGTH=5

def initialize_smartwatch_model(device):
    model=SWNET(in_channels=3,out_len=DATA_FPS*CLIP_LENGTH)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    model.to(device)

    criterion.to(device)


    return model, optimizer, criterion, scheduler


def initialize_new_smartwatch_model(device):
# Example training loop
    model     = TCNDepthNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model, optimizer, criterion, None


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
    avg_depths, min_depth_list, n_cpr = get_avg_depth(accel_mag)


    return avg_depths,min_depth_list, n_cpr

def smartwatch_depth_estimator(smartwatch_data, depth_sensor_data, model, optimizer, criterion):
    """
    This function is a placeholder for the smartwatch depth estimator.
    It is intended to be used in the context of CPR quality assessment.
    """
    
    data=smartwatch_data
    depth_gt=depth_sensor_data
    depth_gt_mask=depth_gt>0

    #get peaks and valleys from GT depth sensor data
    # print("GT Depths:", depth_gt)

    avg_depths_list,min_depths_list,gt_cpr_rate= get_gt_cpr_depth(depth_gt)


    comp_depth=avg_depths_list/MAX_DEPTH

    #normnalize depth
    depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH

    #normalize acceleration

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = data.to(device)
    depth_gt_norm = depth_gt_norm.to(device)

    MIN_ACC=torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0)
    MAX_ACC=torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0)

    MIN_ACC = MIN_ACC.to(device)
    MAX_ACC = MAX_ACC.to(device)

    data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
    # print(data_norm[0])



    model=model.to(device)
    criterion=criterion.to(device)

    comp_depth = comp_depth.to(device)

    rec,depth_pred =model(data_norm.permute(0,2,1))
    print("Smartwatch based depth prediction:", depth_pred.mean().item())
    print("Ground truth depth:", depth_gt_norm.mean().item())
    
    rec_loss=criterion(rec[depth_gt_mask.squeeze()],depth_gt_norm[depth_gt_mask])
    d_loss=criterion(depth_pred,comp_depth)
    # print("rec_loss: ", rec_loss)
    # print("d_loss: ", d_loss)
    loss=d_loss + rec_loss
    print("Total loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # scale depth prediction back to original range
    depth_pred = depth_pred * (MAX_DEPTH)

    return depth_pred
    

def smartwatch_depth_estimator_inference(smartwatch_data, depth_sensor_data, model, optimizer, criterion):
    """
    This function is a placeholder for the smartwatch depth estimator.
    It is intended to be used in the context of CPR quality assessment.
    """
    
    data=smartwatch_data
    depth_gt=depth_sensor_data
    depth_gt_mask=depth_gt>0

    #get peaks and valleys from GT depth sensor data
    # print("GT Depths:", depth_gt)

    avg_depths_list,min_depths_list,gt_cpr_rate= get_gt_cpr_depth(depth_gt)


    comp_depth=avg_depths_list/MAX_DEPTH

    #normnalize depth
    depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH

    #normalize acceleration

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = data.to(device)
    depth_gt_norm = depth_gt_norm.to(device)

    MIN_ACC=torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0)
    MAX_ACC=torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0)

    MIN_ACC = MIN_ACC.to(device)
    MAX_ACC = MAX_ACC.to(device)

    data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
    # print(data_norm[0])

    model=model.to(device)
    comp_depth = comp_depth.to(device)

    rec,depth_pred =model(data_norm.permute(0,2,1))
    print("Smartwatch based depth prediction:", depth_pred.mean().item())
    print("Ground truth depth:", depth_gt_norm.mean().item())
    
    # print("rec_loss: ", rec_loss)
    # print("d_loss: ", d_loss)
    # scale depth prediction back to original range
    depth_pred = depth_pred * (MAX_DEPTH)

    return depth_pred
    

