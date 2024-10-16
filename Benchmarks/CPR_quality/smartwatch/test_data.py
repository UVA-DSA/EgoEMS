import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset
import torch
from torch.utils.data import DataLoader


annot_path=r'Annotations/main_annotation.json'
data_path='C:/Users/lahir/Downloads/data/'

data = EgoExoEMSDataset(annotation_file=annot_path,
                        data_base_path=data_path,
                        fps=30,
                        frames_per_clip=30*5,
                        data_types=['smartwatch'],split='validation')
data_loader = DataLoader(data, batch_size=4, shuffle=True)

for i, batch in enumerate(data_loader):
    data=batch['smartwatch']
    pass



