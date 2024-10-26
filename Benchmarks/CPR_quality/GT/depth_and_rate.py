'''
CPR compression depth and rate calculation from GT depth sensor data
'''

import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader
from Tools.depth_sensor_processing import tools as depth_tools
import numpy as np

DATA_FPS=30
bs=1
CLIP_LENGTH=5
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
log_path=r'Benchmarks/CPR_quality/GT/log.txt'
data_path=r'C:/Users/lahir/Downloads/data/'

data = EgoExoEMSDataset(annotation_file=annot_path,
                        data_base_path=data_path,
                        fps=DATA_FPS,
                        frames_per_clip=DATA_FPS*CLIP_LENGTH,
                        data_types=['smartwatch','depth_sensor'],
                        split='all',
                        activity='chest_compressions',
                        split_path=split_path)

data_loader = DataLoader(data, batch_size=bs, shuffle=True)

if os.path.exists(log_path):
    os.remove(log_path)

CPR_quality={}
for j in range(10):
    for i, batch in enumerate(data_loader):
        depth_gt=batch['depth_sensor'].squeeze()
        p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(depth_gt.detach().numpy(),show=False)
        peak_depths=depth_gt[p]
        valley_depths=depth_gt[v]
        l=min(len(peak_depths),len(valley_depths))
        mean_depth=(peak_depths[:l]-valley_depths[:l]).mean().item()
        CPR_rate=(len(p)+len(v))*0.5/CLIP_LENGTH*60
        subj=batch['subject_id'][0]

        if CPR_rate>190 or CPR_rate<40 or mean_depth>50 or mean_depth>70 or mean_depth<10:
            continue

        print(f'subject {subj}, mean depth: {mean_depth}, CPR rate: {CPR_rate}')

        if subj not in CPR_quality:
            d_={
                'mean_depth':[mean_depth],
                'CPR_rate':[CPR_rate],
            }
            CPR_quality[subj]=d_
        else:
            CPR_quality[subj]['mean_depth'].append(mean_depth)
            CPR_quality[subj]['CPR_rate'].append(CPR_rate)

#print the results
with open(log_path,'w') as f:
    for subj in CPR_quality:
        mean_d=np.mean(CPR_quality[subj]['mean_depth'])
        std_d=np.std(CPR_quality[subj]['mean_depth'])
        mean_r=np.mean(CPR_quality[subj]['CPR_rate'])
        std_r=np.std(CPR_quality[subj]['CPR_rate'])
        f.write(f'subject {subj} mean depth: {mean_d}, std depth: {std_d}, mean CPR rate: {mean_r}, std CPR rate: {std_r}\n')