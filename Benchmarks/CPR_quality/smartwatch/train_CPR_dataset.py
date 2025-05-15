import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import SW_dataset
from torch.utils.data import DataLoader
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Benchmarks.CPR_quality.smartwatch import SWnet
import torch
import utils
from Tools.depth_sensor_processing import tools as depth_tools

data_path=r'D:\CPR_extracted\smartwatch_dataset'
log_path=r'Benchmarks/CPR_quality/GT/log_CPR_data.txt'

EPOCHS=20
BS=16

conf={}
conf['data_root']=data_path
conf['smartwatch']={
    'train_part':  [0,1,2,3,4,5,6,7,8,9,10,11],
    'test_part':  [12,13,14,15,16,17,18,19,20,21,22,23],
    'sw_min': [-80.35959311142295,-79.25364161478956,-80.35959311142295,-80.35959311142295,-79.25364161478956,-80.35959311142295,-80.35959311142295,-79.25364161478956,-80.35959311142295],
    'sw_max' : [80.22816855691997,78.6352995737251,80.22816855691997,80.22816855691997,78.6352995737251,80.22816855691997,80.22816855691997,78.6352995737251,80.22816855691997],
  'depth_min': 8.0,
  'depth_max': 61,
  'n_comp_min': 3.0,
  'n_comp_max': 36.0,
  'normalize': True
}

train_dataset = SW_dataset(conf,'train')
train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

test_dataset = SW_dataset(conf,'test')
test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

model=SWnet.SWNET(in_channels=3,out_len=300)
model=model.double()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')
depth_loss_meter = utils.AverageMeter('depthLoss', ':.4e')


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

def validate():
    freq_meter = utils.AverageMeter('freqError', ':.4e')
    depth_meter = utils.AverageMeter('depthError', ':.4e') 

    for i, batch in enumerate(test_dataloader):
        print(f'Eval, {i}/{len(train_dataloader)} is done',end='\r')
        sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
        sw_data=sw_data[:,0:3,:]
        rec,depth_pred=model(sw_data)
        depth_max=conf['smartwatch']['depth_max']
        depth_min=conf['smartwatch']['depth_min']
        depth_pred=depth_pred*(depth_max-depth_min)+depth_min
        gt_depth=gt_depth*(depth_max-depth_min)+depth_min

        depth_error=torch.sqrt(torch.mean(torch.square(depth_pred-gt_depth.squeeze()))).item()
        _,_,n_cpr_pred=get_avg_depth(rec)
        cpr_error=torch.sqrt(torch.mean(torch.square(gt_n_comp-n_cpr_pred)))

        freq_meter.update(cpr_error.item(),BS)
        depth_meter.update(depth_error,BS)

    print(f'Validation depth error: {depth_meter.avg} , CPR rate error: {freq_meter.avg}')

def train():
    for epoch in range(EPOCHS):
        rec_loss_meter.reset()
        depth_loss_meter.reset()
        for i, batch in enumerate(train_dataloader):
            print(f'Training Epoch {epoch} , {i}/{len(train_dataloader)} is done',end='\r')
            sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
            sw_data=sw_data[:,0:3,:]
            rec,depth_pred=model(sw_data)
            rec_loss=criterion(rec,gt)
            d_loss=criterion(depth_pred,gt_depth.squeeze())
            loss=rec_loss+d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_meter.update(rec_loss.item(),BS)
            depth_loss_meter.update(d_loss.item(),BS)

        scheduler.step()
        print(f'Epoch {epoch} , rec loss: {rec_loss_meter.avg} , depth loss: {depth_loss_meter.avg}')
        validate()


if __name__ == "__main__":
    train()
    # torch.save(model.state_dict(), 'model.pth')
