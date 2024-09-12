import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision import transforms
from collections import OrderedDict
import itertools

transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def collate_fn(batch):
    # Get max sequence length for each modality
    max_video_len = max([clip['frames'].shape[0] for clip in batch])
    max_audio_len = max([clip['audio'].shape[0] for clip in batch])
    max_flow_len = max([clip['flow'].shape[0] for clip in batch])
    max_rgb_len = max([clip['rgb'].shape[0] for clip in batch])

    padded_clips = []
    padded_audio_clips = []
    padded_flow_clips = []
    padded_rgb_clips = []
    keystep_labels = []
    keystep_ids = []
    start_frames = []
    end_frames = []
    start_ts = []
    end_ts = []
    subject_ids = []
    trial_ids = []

    for b in batch:
        video_clip = b['frames']
        audio_clip = b['audio']
        flow_clip = b['flow']
        rgb_clip = b['rgb']
        label = b['keystep_label']
        keystep_id = b['keystep_id']
        start_frame = b['start_frame']
        end_frame = b['end_frame']
        start_t = b['start_t']
        end_t = b['end_t']
        subject_id = b['subject_id']
        trial_id = b['trial_id']

        # Pad video frames
        video_pad_size = max_video_len - video_clip.shape[0]
        if video_pad_size > 0:
            pad = torch.zeros((video_pad_size, *video_clip.shape[1:]))
            video_clip = torch.cat([video_clip, pad], dim=0)

        # Pad audio
        audio_pad_size = max_audio_len - audio_clip.shape[0]
        if audio_pad_size > 0:
            audio_pad = torch.zeros((audio_pad_size, *audio_clip.shape[1:]))
            audio_clip = torch.cat([audio_clip, audio_pad], dim=0)

        # Pad flow data
        flow_pad_size = max_flow_len - flow_clip.shape[0]
        if flow_pad_size > 0:
            flow_pad = torch.zeros((flow_pad_size, *flow_clip.shape[1:]))
            flow_clip = torch.cat([flow_clip, flow_pad], dim=0)

        # Pad rgb data
        rgb_pad_size = max_rgb_len - rgb_clip.shape[0]
        if rgb_pad_size > 0:
            rgb_pad = torch.zeros((rgb_pad_size, *rgb_clip.shape[1:]))
            rgb_clip = torch.cat([rgb_clip, rgb_pad], dim=0)


        padded_clips.append(video_clip)
        padded_audio_clips.append(audio_clip)
        padded_flow_clips.append(flow_clip)
        padded_rgb_clips.append(rgb_clip)
        keystep_labels.append(label)
        keystep_ids.append(keystep_id)
        start_frames.append(start_frame)
        end_frames.append(end_frame)
        start_ts.append(start_t)
        end_ts.append(end_t)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)

    padded_clips = torch.stack(padded_clips)
    padded_audio_clips = torch.stack(padded_audio_clips)
    padded_flow_clips = torch.stack(padded_flow_clips)
    padded_rgb_clips = torch.stack(padded_rgb_clips)
    keystep_ids = torch.tensor(keystep_ids)
    start_frames = torch.tensor(start_frames)
    end_frames = torch.tensor(end_frames)
    start_ts = torch.tensor(start_ts)
    end_ts = torch.tensor(end_ts)
    

    return {
        'frames': padded_clips,
        'audio': padded_audio_clips,
        'flow': padded_flow_clips,
        'rgb': padded_rgb_clips,
        'keystep_label': keystep_labels,
        'keystep_id': keystep_ids,
        'start_frame': start_frames,
        'end_frame': end_frames,
        'start_t': start_ts,
        'end_t': end_ts,
        'subject_id': subject_ids,
        'trial_id': trial_ids
    }

class EgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, data_base_path, fps, frames_per_clip=None, transform=None, audio_sample_rate=16000):
        self.annotation_file = annotation_file
        self.data_base_path = data_base_path
        self.fps = fps
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.data = []

        self._load_annotations()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for subject in annotations['subjects']:
            for trial in subject['trials']:
                avail_streams = trial['streams']
                
                video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                
                if video_path and flow_path and rgb_path:
                    keysteps = trial['keysteps']
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.ceil(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']
                        
                        self.data.append({
                            'video_path': os.path.join(self.data_base_path, video_path),
                            'flow_path': os.path.join(self.data_base_path, flow_path),
                            'rgb_path': os.path.join(self.data_base_path, rgb_path),
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'start_t': step['start_t'],
                            'end_t': step['end_t'],
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
        flow_path = item['flow_path']
        rgb_path = item['rgb_path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        start_t = item['start_t']
        end_t = item['end_t']
        keystep_label = item['keystep_label']
        keystep_id = item['keystep_id']
        subject_id = item['subject']
        trial_id = item['trial']
        
        #print above variables to debug line by line
        print("video_path: ", video_path)
        print("flow_path: ", flow_path)
        print("rgb_path: ", rgb_path)
        print("start_frame: ", start_frame)
        print("end_frame: ", end_frame)
        
        # Load video
        video_reader = VideoReader(video_path, "video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_t, video_reader.seek(start_t)):
            img_tensor = transform(frame['data'])
            frames.append(img_tensor)
        frames = torch.stack(frames)

        # Load audio
        audio_reader = VideoReader(video_path, "audio")
        audio = []
        for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
            audio.append(audio_frame['data'])
        audio = torch.cat(audio, dim=0) if audio else torch.zeros(1, 0)

        # Load flow data
        flow = torch.from_numpy(np.load(flow_path))
        # read only the frames between start_frame and end_frame
        flow = flow[start_frame:end_frame]

        # Load rgb data
        rgb = torch.from_numpy(np.load(rgb_path))
        rgb = rgb[start_frame:end_frame]


        output = {
            'frames': frames,
            'audio': audio,
            'flow': flow,
            'rgb': rgb,
            'keystep_label': keystep_label,
            'keystep_id': keystep_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_t': start_t,
            'end_t': end_t,
            'subject_id': subject_id,
            'trial_id': trial_id
        }
        
        return output

