import os
import sys
import json
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader
from Benchmarks.CPR_quality.smartwatch import SWnet
import utils
from Tools.depth_sensor_processing import tools as depth_tools
import numpy as np

DATA_FPS=30
bs=1
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
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg+'\n')

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

    for i, batch in enumerate(data_loader):
        print("-----------------")
        print(f"Processing validation batch {i}/{len(data_loader)}")

        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0

        # print("data shape: ", data.shape)
        # print("depth_gt shape: ", depth_gt.shape)

        avg_depths_list,min_depths_list,n_cpr=get_avg_depth(depth_gt)
        data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
        rec,depth_pred=model(data_norm.permute(0,2,1))

        depth_pred=depth_pred*MAX_DEPTH

        #average depth error
        avg_depth_error=torch.mean((avg_depths_list-depth_pred)**2)**0.5
        depth_loss_meter.update(avg_depth_error.item(),bs)

        #cpr frequency error
        _,_,n_cpr_pred=get_avg_depth(rec)

        # print("gt_cpr_rate: ", n_cpr)
        # print("pred_cpr_rate: ",n_cpr_pred)

        cpr_error=torch.mean((n_cpr-n_cpr_pred)**2)**0.5
        # print("cpr_error: ", cpr_error)
        
        ncpr_error_meter.update(cpr_error.item(),bs)

        subject = batch['subject_id']
        trial = batch['trial_id']


        accel_magnitude = torch.sqrt(torch.sum(data ** 2, axis=2))
        # apply low pass filter
        low_pass_accel_magnitude = depth_tools.low_pass_filter(accel_magnitude)

        # no filtering
        try:
            avg_depths_list, min_depths_list, keshara_pred_cpr_rate = get_avg_depth(accel_magnitude)
        except:
            print("Error in calculating cpr rate for no filtering")
            keshara_pred_cpr_rate = torch.tensor([0.0])
        

        # low pass filtering
        try:
            avg_depths_list, min_depths_list, low_pass_keshara_pred_cpr_rate = get_avg_depth(low_pass_accel_magnitude)
        except:
            print("Error in calculating cpr rate for low pass filtering")
            low_pass_keshara_pred_cpr_rate = torch.tensor([0.0])

        msg = f'{subject},{trial},GT_Depth:{avg_depths_list.tolist()},Pred_Depth:{depth_pred.tolist()},Depth_error:{avg_depth_error:.2f}mm,GT_CPR_rate:{n_cpr.tolist()},Pred_CPR_rate:{n_cpr_pred.tolist()},CPR_rate_error:{cpr_error/(CLIP_LENGTH)*60:.2f}cpr/min,Keshara_Pred_CPR_rate:{keshara_pred_cpr_rate.tolist()},Low_Pass_Keshara_Pred_CPR_rate:{low_pass_keshara_pred_cpr_rate.tolist()}'
        print(msg)
        write_log_line(log_path,msg)
        print("-----------------")

    msg=f'Validation depth loss: {depth_loss_meter.avg:.2f} mm , CPR rate error: {ncpr_error_meter.avg/(CLIP_LENGTH*DATA_FPS)*60:.2f} cpr/min'
    print(msg)
    write_log_line(log_path,msg)


if __name__ == "__main__":
    
    
        
    # initialize paths
    split_path = split_paths[0]
    
    split = split_path.split('/')[-1].split('.')[0]
    
    log_path = os.path.join(log_base_path, f'debug_validate_log_{split}.txt')
    
    init_log(log_path)
    
    data = EgoExoEMSDataset(annotation_file=annot_path,
                            data_base_path=data_path,
                            fps=DATA_FPS,
                            frames_per_clip=DATA_FPS*CLIP_LENGTH,
                            data_types=['smartwatch','depth_sensor'],
                            split='validation',
                            activity='chest_compressions',
                            split_path=split_path)

    valid_data_loader = DataLoader(data, batch_size=bs, shuffle=False)

    model=SWnet.SWNET(in_channels=3,out_len=DATA_FPS*CLIP_LENGTH)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')
    depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')

        
    # Load the model
    checkpoint_path = f"Benchmarks/CPR_quality/smartwatch/checkpoints/model_{split}.pth"
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    validate(model=model,data_loader=valid_data_loader)







