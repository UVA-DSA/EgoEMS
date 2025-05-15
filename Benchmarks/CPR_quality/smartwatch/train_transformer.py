import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader
from Benchmarks.CPR_quality.smartwatch import SWnet
import utils
from Tools.depth_sensor_processing import tools as depth_tools

DATA_FPS=30
bs=2
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
    for i, batch in enumerate(data_loader):
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0

        avg_depths_list,min_depths_list,n_cpr=get_avg_depth(depth_gt)
        data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)

        gt_cpr_rate = n_cpr

        # run inference
        rec,depth_pred, cpr_rate_pred=model(data_norm.permute(0,2,1))

        depth_pred=depth_pred*MAX_DEPTH

        #average depth error
        avg_depth_error=torch.mean((avg_depths_list-depth_pred)**2)**0.5
        depth_loss_meter.update(avg_depth_error.item(),bs)

        #cpr frequency error
        _,_,n_cpr_pred=get_avg_depth(rec)
        cpr_error=torch.mean((n_cpr-n_cpr_pred)**2)**0.5
        ncpr_error_meter.update(cpr_error.item(),bs)

        subject = batch['subject_id']
        trial = batch['trial_id']

        print(f"{subject},{trial},GT Rate: {gt_cpr_rate}, Pred Rate: {cpr_rate_pred}")
        # msg = f'{subject},{trial},GT_Depth:{avg_depths_list.tolist()},Pred_Depth:{depth_pred.tolist()},Depth_error:{avg_depth_error:.2f}mm,GT_CPR_rate:{n_cpr.tolist()},Pred_CPR_rate:{n_cpr_pred.tolist()},CPR_rate_error:{cpr_error/(DATA_FPS*CLIP_LENGTH)*60:.2f}cpr/min'
        # write_log_line(log_path,msg)

    msg=f'Validation depth loss: {depth_loss_meter.avg:.2f} mm , CPR rate error: {ncpr_error_meter.avg/(CLIP_LENGTH)*60:.2f} cpr/min'
    print(msg)
    write_log_line(log_path,msg)

def train(model, train_data_loader, valid_data_loader, criterion, optimizer, scheduler, log_path, model_save_path):
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_data_loader):
            print(f'Epoch {epoch} , {i}/{len(train_data_loader)} is done',end='\r')
            data=batch['smartwatch'].float()
            depth_gt=batch['depth_sensor'].squeeze()

            #get peaks and valleys from GT depth sensor data
            avg_depths_list,min_depths_list,gt_cpr_rate=get_avg_depth(depth_gt)

            # Compute mean and standard deviation along the `samples` dimension for each channel
            mean = data.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1, 3]
            std = data.std(dim=1, keepdim=True)     # Shape: [batch_size, 1, 3]
            # Standardize each channel independently across the samples dimension
            data_normalized = (data - mean) / (std + 1e-6)  # Adding epsilon to avoid division by zero
            sw_data = data_normalized

            # run inference
            pred_cpr_rate = model(sw_data)

            # Compute loss
            cpr_rate_loss = criterion(pred_cpr_rate, gt_cpr_rate)


            print("cpr_rate_pred: ", pred_cpr_rate)
            print("cpr_rate_gt: ", gt_cpr_rate)
            # depth_gt_mask=depth_gt>0

            # comp_depth=avg_depths_list/MAX_DEPTH

            # #normnalize depth
            # depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH

            # #normalize acceleration
            # data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
            # # print(data_norm.shape)
            # # print(data_norm[0])

            # rec,depth_pred, cpr_rate_pred =model(data_norm.permute(0,2,1))

            # # print("cpr_rate_pred: ", cpr_rate_pred)
            # cpr_rate_loss = criterion(cpr_rate_pred, gt_cpr_rate)
            
            # rec_loss=criterion(rec[depth_gt_mask],depth_gt_norm[depth_gt_mask])
            # d_loss=criterion(depth_pred,comp_depth)
            # # print("rec_loss: ", rec_loss)
            # # print("d_loss: ", d_loss)
            # loss=rec_loss+d_loss+cpr_rate_loss
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # rec_loss_meter.update(rec_loss.item(),bs)
            # depth_loss_meter.update(d_loss.item(),bs)
            cpr_rate_loss_meter.update(cpr_rate_loss.item(),bs)

        msg=f'Training Epoch {epoch} , rec loss: {rec_loss_meter.avg} , depth loss: {depth_loss_meter.avg}, cpr rate loss: {cpr_rate_loss_meter.avg}'
        print(msg)
        write_log_line(log_path,msg)
        scheduler.step()

        if (epoch+1)%1==0:
            validate(model,valid_data_loader)
            torch.save(model.state_dict(),model_save_path)

if __name__ == "__main__":
    
        # initialize paths
    split_path = split_paths[3]
    
    split = split_path.split('/')[-1].split('.')[0]
    
    log_path = os.path.join(log_base_path, f'debug_train_log_{split}.txt')
    
    model_save_path = os.path.join(model_save_base_path, f'debug_model_{split}.pth')
    
    init_log(log_path)
    
    data = EgoExoEMSDataset(annotation_file=annot_path,
                        data_base_path="",
                        fps=DATA_FPS,
                        frames_per_clip=DATA_FPS*CLIP_LENGTH,
                        data_types=['smartwatch','depth_sensor'],
                        split='train',
                        activity='chest_compressions',
                        split_path=split_path)

    train_data_loader = DataLoader(data, batch_size=bs, shuffle=True)

    data = EgoExoEMSDataset(annotation_file=annot_path,
                            data_base_path=data_path,
                            fps=DATA_FPS,
                            frames_per_clip=DATA_FPS*CLIP_LENGTH,
                            data_types=['smartwatch','depth_sensor'],
                            split='validation',
                            activity='chest_compressions',
                            split_path=split_path)

    valid_data_loader = DataLoader(data, batch_size=bs, shuffle=True)

    model = SWnet.TransformerCPR()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')
    depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')
    cpr_rate_loss_meter = utils.AverageMeter('cprRateLoss', ':.4e')
    
    train(model=model,train_data_loader=train_data_loader,valid_data_loader=valid_data_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, log_path=log_path, model_save_path=model_save_path)