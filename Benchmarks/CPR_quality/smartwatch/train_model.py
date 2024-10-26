import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader
from Benchmarks.CPR_quality.smartwatch import SWnet
import utils
from Tools.depth_sensor_processing import tools as depth_tools

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

annot_path=r'Annotations/main_annotation.json'
split_path=r'Annotations/splits/cpr_quality/subject_splits.json'
log_path=r'Benchmarks/CPR_quality/smartwatch/log.txt'

#set these paths to your own paths
model_save_path=r'C:/Users/lahir/Downloads/data/model.pth'
data_path=r'C:/Users/lahir/Downloads/data/'


data = EgoExoEMSDataset(annotation_file=annot_path,
                        data_base_path=data_path,
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

model=SWnet.SWNET(in_channels=3,out_len=DATA_FPS*CLIP_LENGTH)
#load weights
model.load_state_dict(torch.load(model_save_path,weights_only=True))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')
depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')

def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)

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
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0

        avg_depths_list,min_depths_list,n_cpr=get_avg_depth(depth_gt)
        data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
        rec,depth_pred=model(data_norm.permute(0,2,1))

        depth_pred=depth_pred*MAX_DEPTH

        #average depth error
        avg_depth_error=torch.mean((avg_depths_list-depth_pred)**2)**0.5
        depth_loss_meter.update(avg_depth_error.item(),bs)

        #cpr frequency error
        _,_,n_cpr_pred=get_avg_depth(rec)
        cpr_error=torch.mean((n_cpr-n_cpr_pred)**2)**0.5
        ncpr_error_meter.update(cpr_error.item(),bs)

    msg=f'Validation depth loss: {depth_loss_meter.avg:.2f} mm , CPR rate error: {ncpr_error_meter.avg/(DATA_FPS*CLIP_LENGTH)*60:.2f} cpr/min'
    print(msg)
    write_log_line(log_path,msg)

init_log(log_path)
def train():
    init_log(log_path)
    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_data_loader):
            print(f'Epoch {epoch} , {i}/{len(train_data_loader)} is done',end='\r')
            data=batch['smartwatch'].float()
            depth_gt=batch['depth_sensor'].squeeze()
            depth_gt_mask=depth_gt>0

            #get peaks and valleys from GT depth sensor data
            avg_depths_list,min_depths_list,_=get_avg_depth(depth_gt)

            comp_depth=avg_depths_list/MAX_DEPTH

            #normnalize depth
            depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH

            #normalize acceleration
            data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)

            rec,depth_pred=model(data_norm.permute(0,2,1))
            rec_loss=criterion(rec[depth_gt_mask],depth_gt_norm[depth_gt_mask])
            d_loss=criterion(depth_pred,comp_depth)
            loss=rec_loss+d_loss

            #delete
            # import matplotlib.pyplot as plt

            # plt.plot(rec[0,:].detach().numpy())
            # plt.show(block=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_meter.update(rec_loss.item(),bs)
            depth_loss_meter.update(d_loss.item(),bs)

        msg=f'Training Epoch {epoch} , rec loss: {rec_loss_meter.avg} , depth loss: {depth_loss_meter.avg}'
        print(msg)
        write_log_line(log_path,msg)
        scheduler.step()

        if (epoch+1)%1==0:
            validate(model,valid_data_loader)
            torch.save(model.state_dict(),model_save_path)

def plot():
    import matplotlib.pyplot as plt

    for i, batch in enumerate(train_data_loader):
        depth_gt=batch['depth_sensor'].squeeze()
        data=batch['smartwatch'].float()
        depth_gt_mask=depth_gt>0
        avg_depths_list,min_depths_list,_=get_avg_depth(depth_gt)
        comp_depth=avg_depths_list/MAX_DEPTH
        depth_gt_norm=(depth_gt-min_depths_list.unsqueeze(1))/MAX_DEPTH
        #normalize acceleration
        data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)
        rec,depth_pred=model(data_norm.permute(0,2,1))

        plt.plot(rec[10,:].detach().numpy())
        plt.show(block=True)
        pass
    
if __name__ == "__main__":
    # train()
    # plot()
    validate(model,valid_data_loader)