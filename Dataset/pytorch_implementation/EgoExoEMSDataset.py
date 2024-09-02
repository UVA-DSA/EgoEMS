import os
import json
import math
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    # Find the max length of clips in the batch
    max_len = max([clip['frames'].shape[0] for clip in batch])

    # Prepare the structure to hold the batch data
    padded_clips = []
    keystep_labels = []
    keystep_ids = []
    start_frames = []
    end_frames = []
    subject_ids = []
    trial_ids = []

    for b in batch:
        clip = b['frames']
        label = b['keystep_label']
        keystep_id = b['keystep_id']
        start_frame = b['start_frame']
        end_frame = b['end_frame']
        subject_id = b['subject_id']
        trial_id = b['trial_id']

        
        pad_size = max_len - clip.shape[0]
        if pad_size > 0:
            # Pad with zeros (assuming RGB, so 3 channels)
            pad = torch.zeros((pad_size, *clip.shape[1:]))
            clip = torch.cat([clip, pad], dim=0)
        padded_clips.append(clip)
        keystep_labels.append(label)
        keystep_ids.append(keystep_id)
        start_frames.append(start_frame)
        end_frames.append(end_frame)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)

    # Stack the padded clips into a tensor
    padded_clips = torch.stack(padded_clips)
    keystep_ids = torch.tensor(keystep_ids)
    start_frames = torch.tensor(start_frames)
    end_frames = torch.tensor(end_frames)
    # subject_ids = torch.tensor(subject_ids)
    # trial_ids = torch.tensor(trial_ids)

    # The batch will be a dictionary, similar to the individual items
    return {
        'frames': padded_clips,
        'keystep_label': keystep_labels,
        'keystep_id': keystep_ids,
        'start_frame': start_frames,
        'end_frame': end_frames,
        'subject_id': subject_ids,
        'trial_id': trial_ids
    }

class EgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, video_base_path, fps, frames_per_clip=None, transform=None):
        self.annotation_file = annotation_file
        self.video_base_path = video_base_path
        self.fps = fps
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.data = []

        self._load_annotations()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for subject in annotations['subjects']:
            for trial in subject['trials']:
                stream = trial['streams'].get('egocam_rgb_audio', None)
                if stream and 'keysteps' in stream:
                    video_path = os.path.join(self.video_base_path, stream['file_path'])
                    keysteps = stream['keysteps']
                    
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.ceil(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        if self.frames_per_clip is None:
                            # If frames_per_clip is not specified, treat the entire keystep as one clip
                            self.data.append({
                                'video_path': video_path,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'keystep_label': label,
                                'keystep_id': keystep_id,
                                'subject': subject['subject_id'],
                                'trial': trial['trial_id'] 
                            })
                        else:
                            # Otherwise, split the keystep into multiple clips
                            clip_count = math.ceil((end_frame - start_frame) / self.frames_per_clip)
                            for i in range(clip_count):
                                clip_start_frame = start_frame + i * self.frames_per_clip
                                clip_end_frame = min(clip_start_frame + self.frames_per_clip, end_frame)

                                self.data.append({
                                    'video_path': video_path,
                                    'start_frame': clip_start_frame,
                                    'end_frame': clip_end_frame,
                                    'keystep_label': label,
                                    'keystep_id': keystep_id,
                                    'subject': subject['subject_id'],
                                    'trial': trial['trial_id'] 
                                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item['video_path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        keystep_label = item['keystep_label']
        keystep_id = item['keystep_id']
        subject_id = item['subject']
        trial_id = item['trial']

        # Load the video frames for the clip
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        

        if self.frames_per_clip is not None and len(frames) < self.frames_per_clip:
            # Handle cases where the clip is shorter than frames_per_clip
            pad_len = self.frames_per_clip - len(frames)
            frames.extend([frames[-1]] * pad_len)  # Pad with the last frame
        
        frames = torch.stack(frames)  # Shape will be [T, C, H, W]

        output = {
            'frames': frames,
            'keystep_label': keystep_label,
            'keystep_id': keystep_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'subject_id': subject_id,
            'trial_id': trial_id
        }
        
        return output





if __name__ == '__main__':
    dataset = EgoExoEMSDataset(annotation_file='../../Annotations/main_annotation.json',
                                 video_base_path='',
                                 fps=30,  transform=transform)

    # Access a sample
    print(len(dataset))

    # create a data loader
    # batch size is 1 for simplicity and to ensure only a full clip related to a key step is given without collating.
    # if batch size is greater than 1, collate_fn will be called to collate the data.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Iterate over the data loader and print the shape of the batch
    for batch in data_loader:
        print(batch.keys())
        print(batch['frames'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'], batch['subject_id'], batch['trial_id'])
        break