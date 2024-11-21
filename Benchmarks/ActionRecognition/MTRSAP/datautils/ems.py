import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torchvision as T


from .video_utils import VideoClips


def scale(x):
    return x / 255.

def reshape(x):
    return x.permute(0, 3, 1, 2)


tfs = T.transforms.Compose([
    T.transforms.Lambda(scale),  # scale in [0, 1]
    T.transforms.Resize((256, 256)),
    # T.transforms.Lambda(reshape),  # reshape to [B, C, H, W]
])

class VideoDataset(Dataset):
    def __init__(self, base_path, fold, skip_frames=0, transform=None, clip_length_in_frames=None, train=True, num_workers=4):
        self.base_path = base_path
        self.fold = fold
        self.skip_frames = skip_frames
        self.transform = transform
        self.clip_length_in_frames = clip_length_in_frames  # Number of frames to sample from each video
        self.data = []
        self.video_list = []
        self.num_workers = num_workers
        self.train = train
        self.load_annotations()
        
        
        self.video_clips = VideoClips(self.video_list, clip_length_in_frames=self.clip_length_in_frames, frames_between_clips=self.skip_frames, num_workers=self.num_workers, transforms=self.transform)
        
    def load_annotations(self):
        if(self.train):
            annotation_file = os.path.join(self.base_path, f'new_train_video_fold_{self.fold}.txt')
        else:
            annotation_file = os.path.join(self.base_path, f'new_test_video_fold_{self.fold}.txt')

        with open(annotation_file, 'r') as file:
            for line in file:
                video_path, label = line.strip().split()
                full_path = os.path.join(self.base_path, video_path)
                self.data.append((full_path, int(label)))
                self.video_list.append(full_path)
                
        print(f"Loaded {len(self.data)} videos from {annotation_file}")
        print(f"Video List: {self.video_list}")

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video_path, label = self.data[video_idx]
        
        return video, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# base_path = '/scratch/anonymous/2024/datasets/EMS_Datasets/Organized/EMS_Interventions/annotations/'


# # # Example usage
# train_dataset = VideoDataset(base_path=base_path, fold=1, skip_frames=25, transform=transform, clip_length_in_frames=30, train=True)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# for videos, labels in train_loader:
#     # Implement training loop or other processing here
#     print(videos.shape, labels[0])
#     # break



