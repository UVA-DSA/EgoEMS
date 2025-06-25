import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)

from EgoExoEMS.EgoExoEMS import  WindowEgoExoEMSDataset, EgoExoEMSDataset, collate_fn, transform, window_collate_fn
import torch
from torch.utils.data import DataLoader
from Benchmarks.CPR_quality.smartwatch import SWnet
import utils
from Tools.depth_sensor_processing import tools as depth_tools
import numpy as np

from Benchmarks.CPR_quality.smartwatch.config import DefaultArgsNamespace

from Benchmarks.CPR_quality.smartwatch.utils import *

DATA_FPS=30
bs=16
CLIP_LENGTH=5
EPOCHS=10
# utils.get_data_stats(data_loader)
'''
data stats:
max depth: 82.0
min depth: 0.0
max acc: tensor([69.7335, 59.6060, 77.9001])
min acc: tensor([-49.3732, -77.9384, -49.8976])
'''
MIN_ACC=torch.tensor([-49.3732, -77.9384, -49.8976]).unsqueeze(0).unsqueeze(0)
MAX_ACC=torch.tensor([69.7335, 59.6060, 77.9001]).unsqueeze(0).unsqueeze(0)
MIN_DEPTH=0.0
MAX_DEPTH=82.0

annot_path=r'Annotations/main_annotation_cpr_quality.json'
split_paths = [r'Annotations/splits/cpr_quality/split_1.json', r'Annotations/splits/cpr_quality/split_2.json', r'Annotations/splits/cpr_quality/split_3.json', r'Annotations/splits/cpr_quality/split_4.json']

log_base_path=r'Benchmarks/CPR_quality/smartwatch/logs/'

#set these paths to your own paths
model_save_base_path = r'Benchmarks/CPR_quality/smartwatch/checkpoints/'
model_save_path = r''
data_path = r''

def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
        # create the log folder if it does not exist
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg+'\n')



def get_gt_cpr_depth(gt_window):
    # 1) Turn input into a (1, N, 3) torch tensor

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

def get_avg_depth(depth_gt):
    avg_depth_list,min_depth_list,n_cpr=[],[],[]
    if len(depth_gt.shape)==1:
        depth_gt=depth_gt.unsqueeze(0)
    for i in range(depth_gt.shape[0]):
        p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(depth_gt[i,:].detach().numpy(),show=False)
        peak_depths=depth_gt[i,p]
        valley_depths=depth_gt[i,v]
        l=min(len(peak_depths),len(valley_depths))
        avg_depth_list.append((peak_depths[:l]-valley_depths[:l]).mean().item())
        min_depth_list.append(valley_depths[:l].min().item())
        n_cpr.append((len(p)+len(v))*0.5)
    return torch.tensor(avg_depth_list),torch.tensor(min_depth_list),torch.tensor(n_cpr)


def validate(model,data_loader):
    depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')
    ncpr_error_meter = utils.AverageMeter('cprError', ':.4e')
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, batch in enumerate(data_loader):
        print("**" * 20)

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

        smartwatch = input['smartwatch'][0]    # [T,3]
        gt_sensor_data = gt_sensor_data[0]     # [T,3]

        total = smartwatch.shape[0]
        print(f"Processing video {video_id} with {total} frames, smartwatch data length {len(smartwatch)}, and gt data length {len(gt_sensor_data)}")

        window_size = DATA_FPS * CLIP_LENGTH
        print(f"Window size: {window_size} frames")

        for start in range(0, total, window_size):
            end = start + window_size
            if end > total:
                break

            data= smartwatch[start:end]
            depth_gt=gt_sensor_data[start:end]
            depth_gt_mask=depth_gt>0

            avg_depths_list,min_depths_list,n_cpr= get_gt_cpr_depth(depth_gt)

            data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)

            gt_cpr_rate = n_cpr

            # run inference
            rec,depth_pred=model(data_norm.permute(0,2,1))

            depth_pred=depth_pred*MAX_DEPTH

            #average depth error
            avg_depth_error=torch.mean((avg_depths_list-depth_pred)**2)**0.5
            depth_loss_meter.update(avg_depth_error.item(),bs)

            #cpr frequency error
            _,_,n_cpr_pred=get_avg_depth(rec)
            cpr_error=torch.mean((n_cpr-n_cpr_pred)**2)**0.5
            ncpr_error_meter.update(cpr_error.item(),bs)



            # 1) Turn input into a (1, N, 3) torch tensor
            if not torch.is_tensor(data):
                data = torch.tensor(data, dtype=torch.float32)
            else:
                data = data.float()
            if data.ndim == 2:             # (N,3) → (1,N,3)
                data = data.unsqueeze(0)
            elif data.ndim == 3 and data.shape[0] != 1:
                raise ValueError("Expected window shape (N,3) or (1,N,3), got %s" % (data.shape,))

            # 2) Compute per‑sample magnitude → shape (1, N)
            accel_magnitude = torch.norm(data, dim=2)
            # accel_magnitude = torch.sqrt(torch.sum(data ** 2, axis=2))
            accel_magnitude_np = accel_magnitude.numpy().flatten()
            timestamps = np.linspace(0, len(accel_magnitude_np) / DATA_FPS, len(accel_magnitude_np))

                    
            avg_depths_list, min_depths_list, pred_cpr_rate = get_avg_depth(accel_magnitude)
            
            subject = batch['subject_id']
            trial = batch['trial_id']

            print(f"{subject},{trial},GT Rate:{gt_cpr_rate.tolist()},Lahiru Pred Rate:{n_cpr_pred.tolist()},Keshara Pred Rate:{pred_cpr_rate}")
            # msg = f'{subject},{trial},GT_Depth:{avg_depths_list.tolist()},Pred_Depth:{depth_pred.tolist()},Depth_error:{avg_depth_error:.2f}mm,GT_CPR_rate:{n_cpr.tolist()},Pred_CPR_rate:{n_cpr_pred.tolist()},CPR_rate_error:{cpr_error/(DATA_FPS*CLIP_LENGTH)*60:.2f}cpr/min'
            # write_log_line(log_path,msg)

        msg=f'Validation depth loss: {depth_loss_meter.avg:.2f} mm , CPR rate error: {ncpr_error_meter.avg/(CLIP_LENGTH)*60:.2f} cpr/min'
        print(msg)
        write_log_line(log_path,msg)

def train(model, train_data_loader, valid_data_loader, criterion, optimizer, scheduler, log_path, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_data_loader):

            print("**" * 20)
            print(f'Epoch {epoch} , {i}/{len(train_data_loader)} is done',end='\r')

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

            smartwatch = input['smartwatch'][0]    # [T,3]
            gt_sensor_data = gt_sensor_data[0]     # [T,3]

            total = smartwatch.shape[0]
            print(f"Processing video {video_id} with {total} frames, smartwatch data length {len(smartwatch)}, and gt data length {len(gt_sensor_data)}")

            window_size = DATA_FPS * CLIP_LENGTH
            print(f"Window size: {window_size} frames")

            for start in range(0, total, window_size):
                end = start + window_size
                if end > total:
                    break

                data= smartwatch[start:end]
                depth_gt=gt_sensor_data[start:end]
                depth_gt_mask=depth_gt>0


                # print("Data shape: ", data.shape)
                # print("Depth GT shape: ", depth_gt.shape)
                # print("Depth GT Mask shape: ", depth_gt_mask.shape)

                #get peaks and valleys from GT depth sensor data
                # avg_depths_list,min_depths_list,gt_cpr_rate=get_avg_depth(depth_gt)
                avg_depths_list,min_depths_list,gt_cpr_rate= get_gt_cpr_depth(depth_gt)


                comp_depth=avg_depths_list/MAX_DEPTH

                #normnalize depth
                depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH

                #normalize acceleration
                data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
                # print(data_norm[0])

                rec,depth_pred =model(data_norm.permute(0,2,1))


                # print("rec shape: ", rec.shape)   
                # print("Smartwatch based depth prediction shape:", depth_pred.shape)
                # print("depth_gt_norm shape:", depth_gt_norm.shape)
                # print(f"depth_gt_norm[depth_gt_mask] : {depth_gt_norm[depth_gt_mask]}")

                
                rec_loss=criterion(rec[depth_gt_mask.squeeze()],depth_gt_norm[depth_gt_mask])
                d_loss=criterion(depth_pred,comp_depth)
                # print("rec_loss: ", rec_loss)
                # print("d_loss: ", d_loss)
                loss=rec_loss+d_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rec_loss_meter.update(rec_loss.item(),bs)
                depth_loss_meter.update(d_loss.item(),bs)

            print(f"CPR ground truth depth: {comp_depth.mean().item():.2f} mm, GT CPR rate: {gt_cpr_rate.mean().item():.2f} cpr/min")
            print(f"Predicted depth: {depth_pred.mean().item():.2f} mm")
            print(f"Epoch {epoch}, Batch {i}, Rec Loss: {rec_loss_meter.avg:.4f}, Depth Loss: {depth_loss_meter.avg:.4f}", end='\r')

        msg=f'Training Epoch {epoch} , rec loss: {rec_loss_meter.avg} , depth loss: {depth_loss_meter.avg}, cpr rate loss: {cpr_rate_loss_meter.avg}'
        print(msg)
        write_log_line(log_path,msg)
        scheduler.step()

        if (epoch+1)%1==0:
            validate(model,valid_data_loader)
            
            torch.save(model.state_dict(),model_save_path)

if __name__ == "__main__":

    args = DefaultArgsNamespace()
    
    log_path = os.path.join(log_base_path, f'debug_train_log.txt')
    model_save_path = os.path.join(model_save_base_path, f'debug_model.pth')

    # create checkpoint directory if it does not exist
    if not os.path.exists(model_save_base_path):
        os.makedirs(model_save_base_path)
        
    
    init_log(log_path)
    
    # prepare dataset & loader

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


    model=SWnet.SWNET(in_channels=3,out_len=DATA_FPS*CLIP_LENGTH)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')
    depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')
    cpr_rate_loss_meter = utils.AverageMeter('cprRateLoss', ':.4e')
    
    train(model=model,train_data_loader=train_loader,valid_data_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, log_path=log_path, model_save_path=model_save_path)