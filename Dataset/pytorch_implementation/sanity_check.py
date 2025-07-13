import os
import torchaudio
import torch
from EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset, collate_fn, transform
import numpy as np
import warnings
from torch.utils.data import  DataLoader

warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


annotation_file = "../../Annotations/aaai26_main_annotation_classification.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

fps = 29.97
observation_window = 150
data_types = [ 'video', 'audio']
task = "classification"  # or "segmentation" or "cpr_quality"

train_dataset = EgoExoEMSDataset(annotation_file=annotation_file,
                                data_base_path='',
                                fps=fps, frames_per_clip=observation_window, transform=transform, data_types=data_types, task=task)


train_class_stats = train_dataset._get_class_stats()
print("Train class stats: ", train_class_stats)
# print number of keys in the dictionary
print("Train Number of classes: ", len(train_class_stats.keys()))

# Create DataLoaders for training and validation subsets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print("Number of batches in train loader:", len(train_loader))

for i, batch in enumerate(train_loader):
    print("-*"*30)
    print(f"Batch {i+1}:")
    
    # Print shapes of the tensors in the batch
    print("frames shape:", batch['frames'].shape)
    print("audio shape:", batch['audio'].shape)
    
    # Print labels and other metadata
    print("Label", batch['keystep_label'])
    print("Start Frames:", batch['start_frame'])
    print("End Frames:", batch['end_frame'])
    print("Trial IDs:", batch['trial_id'])
    print("Subject IDs:", batch['subject_id'])
    
