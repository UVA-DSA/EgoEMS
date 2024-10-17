import os
import torchaudio
import torch
from EgoExoEMS.EgoExoEMS import EgoExoEMSDataset, collate_fn, transform
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")

root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng9/cardiac_arrest/1/GoPro/GX010346_encoded_trimmed.mp4"  # Folder in which all videos lie in a specific structure
annotation_file = "../../Annotations/main_annotation.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

# train_annotation_file = "../../Annotations/splits/keysteps/train_split.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
# val_annotation_file = "../../Annotations/splits/keysteps/val_split.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
# test_annotation_file = "../../Annotations/splits/keysteps/test_split.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

train_dataset = EgoExoEMSDataset(annotation_file=annotation_file,
                                data_base_path='',
                                fps=29.97, frames_per_clip=30, transform=transform, data_types=[ 'video','audio','smartwatch', 'flow', 'rgb', 'depth_sensor'])

# Access a sample
print(len(train_dataset))


# create a data loader
# batch size is 1 for simplicity and to ensure only a full clip related to a key step is given without collating.
# if batch size is greater than 1, collate_fn will be called to collate the data.
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(len(data_loader))

# Iterate over the data loader and print the shape of the batch
for batch in data_loader:
    print(batch['frames'].shape, batch['audio'].shape, batch['flow'].shape, batch['rgb'].shape, batch['smartwatch'].shape, batch['depth_sensor'].shape , batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'],batch['start_t'], batch['end_t'],  batch['subject_id'], batch['trial_id'])
    print("*"*4 + "="*50 + "*"*4)
    
    # audio_tensor = batch['audio'][0]
    # #transpose
    # audio_tensor = audio_tensor.transpose(0,1)
    # torchaudio.save("./visualizations/audio.wav", audio_tensor,48000)
    # break   