import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader
from Benchmarks.CPR_quality.smartwatch import SWnet
import utils

DATA_FPS=30
bs=32
CLIP_LENGTH=5
EPOCHS=100


annot_path=r'Annotations/main_annotation.json'
data_path='C:/Users/lahir/Downloads/data/'
split_path=r'Annotations/splits/cpr_quality/subject_splits.json'

 #in seconds

data = EgoExoEMSDataset(annotation_file=annot_path,
                        data_base_path=data_path,
                        fps=DATA_FPS,
                        frames_per_clip=DATA_FPS*CLIP_LENGTH,
                        data_types=['smartwatch','depth_sensor'],
                        split='train',
                        activity='chest_compressions',
                        split_path=split_path)

train_data_loader = DataLoader(data, batch_size=bs, shuffle=True)

model=SWnet.SWNET(in_channels=3,out_len=DATA_FPS*CLIP_LENGTH)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

rec_loss_meter = utils.AverageMeter('recLoss', ':.4e')

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

for epoch in range(EPOCHS):
    for i, batch in enumerate(train_data_loader):
        print(f'Epoch {epoch} , {i}/{len(train_data_loader)} is done',end='\r')
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0

        #normnalize depth
        depth_gt_min = torch.where(depth_gt_mask, depth_gt, torch.tensor(float('inf')))
        mins=depth_gt_min.min(dim=1).values
        depth_norm=depth_gt-mins.unsqueeze(1)
        depth_norm=(depth_norm-MIN_DEPTH)/(MAX_DEPTH-MIN_DEPTH)
        utils.detect_peaks_and_valleys_depth_sensor(depth_gt[30,:].detach().numpy(),show=True)

        #normalize acceleration
        data_norm=(data-MIN_ACC)/(MAX_ACC-MIN_ACC)

        rec,depth_pred=model(data_norm.permute(0,2,1))
        rec_loss=criterion(rec[depth_gt_mask],depth_norm[depth_gt_mask])
        
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
        rec_loss_meter.update(rec_loss.item(),bs)
    print(rec_loss_meter.avg)






