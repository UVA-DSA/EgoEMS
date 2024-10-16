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
    # Initialize empty lists for all possible modalities
    padded_clips = []
    padded_audio_clips = []
    padded_flow_clips = []
    padded_rgb_clips = []
    padded_smartwatch_clips = []
    keystep_labels = []
    keystep_ids = []
    start_frames = []
    end_frames = []
    start_ts = []
    end_ts = []
    subject_ids = []
    trial_ids = []

    # Initialize max lengths for each modality, check if each is available
    max_video_len = max([clip['frames'].shape[0] for clip in batch if 'frames' in clip], default=0)
    max_audio_len = max([clip['audio'].shape[0] for clip in batch if 'audio' in clip], default=0)
    max_flow_len = max([clip['flow'].shape[0] for clip in batch if 'flow' in clip], default=0)
    max_rgb_len = max([clip['rgb'].shape[0] for clip in batch if 'rgb' in clip], default=0)
    max_smartwatch_len = max([clip['smartwatch'].shape[1] if isinstance(clip['smartwatch'], torch.Tensor) else 0 for clip in batch if 'smartwatch' in clip], default=0)

    for b in batch:
        # Pad video frames if available
        if 'frames' in b:
            video_clip = b['frames']
            video_pad_size = max_video_len - video_clip.shape[0]
            if video_pad_size > 0:
                video_pad = torch.zeros((video_pad_size, *video_clip.shape[1:]))
                video_clip = torch.cat([video_clip, video_pad], dim=0)
            padded_clips.append(video_clip)

        # Pad audio if available
        if 'audio' in b:
            audio_clip = b['audio']
            audio_pad_size = max_audio_len - audio_clip.shape[0]
            if audio_pad_size > 0:
                audio_pad = torch.zeros((audio_pad_size, *audio_clip.shape[1:]))
                audio_clip = torch.cat([audio_clip, audio_pad], dim=0)
            padded_audio_clips.append(audio_clip)

        # Pad flow data if available
        if 'flow' in b:
            flow_clip = b['flow']
            flow_pad_size = max_flow_len - flow_clip.shape[0]
            if flow_pad_size > 0:
                flow_pad = torch.zeros((flow_pad_size, *flow_clip.shape[1:]))
                flow_clip = torch.cat([flow_clip, flow_pad], dim=0)
            padded_flow_clips.append(flow_clip)

        # Pad rgb data if available
        if 'rgb' in b:
            rgb_clip = b['rgb']
            rgb_pad_size = max_rgb_len - rgb_clip.shape[0]
            if rgb_pad_size > 0:
                rgb_pad = torch.zeros((rgb_pad_size, *rgb_clip.shape[1:]))
                rgb_clip = torch.cat([rgb_clip, rgb_pad], dim=0)
            padded_rgb_clips.append(rgb_clip)

        # Pad smartwatch data if available
        if 'smartwatch' in b:
            smartwatch_clip = b['smartwatch']
            smartwatch_pad_size = max_smartwatch_len - smartwatch_clip.shape[1]
            if smartwatch_pad_size > 0:
                smartwatch_pad = torch.zeros((smartwatch_clip.shape[0], smartwatch_pad_size))
                smartwatch_clip = torch.cat([smartwatch_clip, smartwatch_pad], dim=1)
            padded_smartwatch_clips.append(smartwatch_clip)

        # Collect other fields
        keystep_labels.append(b['keystep_label'])
        keystep_ids.append(b['keystep_id'])
        start_frames.append(b['start_frame'])
        end_frames.append(b['end_frame'])
        start_ts.append(b['start_t'])
        end_ts.append(b['end_t'])
        subject_ids.append(b['subject_id'])
        trial_ids.append(b['trial_id'])

    output = {
        'keystep_label': keystep_labels,
        'keystep_id': torch.tensor(keystep_ids),
        'start_frame': torch.tensor(start_frames),
        'end_frame': torch.tensor(end_frames),
        'start_t': torch.tensor(start_ts),
        'end_t': torch.tensor(end_ts),
        'subject_id': subject_ids,
        'trial_id': trial_ids
    }

    # Only include modality data if it exists
    if padded_clips:
        output['frames'] = torch.stack(padded_clips)
    if padded_audio_clips:
        output['audio'] = torch.stack(padded_audio_clips)
    if padded_flow_clips:
        output['flow'] = torch.stack(padded_flow_clips)
    if padded_rgb_clips:
        output['rgb'] = torch.stack(padded_rgb_clips)
    if padded_smartwatch_clips:
        output['smartwatch'] = torch.stack(padded_smartwatch_clips)

    return output


class EgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, data_base_path, fps, 
                frames_per_clip=None, transform=None,
                data_types=['smartwatch'],
                audio_sample_rate=48000):
        
        self.annotation_file = annotation_file
        self.data_base_path = data_base_path
        self.fps = fps
        self.frames_per_clip = frames_per_clip  # Store frames_per_clip
        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.data = []
        self.clip_indices = []  # This will store (item_idx, clip_idx) tuples
        self.data_types=data_types
        
        self._load_annotations()
        self._generate_clip_indices()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for subject in annotations['subjects']:
            for trial in subject['trials']:
                avail_streams = trial['streams']
                
                # Initialize paths to None by default
                video_path = None
                flow_path = None
                rgb_path = None
                smartwatch_path = None

                # Check for each data type and retrieve the corresponding file path
                if 'video' in self.data_types:
                    video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'flow' in self.data_types:
                    flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                if 'rgb' in self.data_types:
                    rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                if 'smartwatch' in self.data_types:
                    smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
   

                # Skip the trial if any required data type is not available
                if ('video' in self.data_types and not video_path) or \
                ('flow' in self.data_types and not flow_path) or \
                ('rgb' in self.data_types and not rgb_path) or \
                ('smartwatch' in self.data_types and not smartwatch_path):
                    print(f"[Warning] Skipping trial {trial['trial_id']} for subject {subject['subject_id']} due to missing data")
                    continue

                if video_path or flow_path or rgb_path or smartwatch_path:
                    keysteps = trial['keysteps']
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.ceil(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        dict={}
                        if 'video' in self.data_types:
                            dict['video_path']=os.path.join(self.data_base_path, video_path)
                        if 'flow' in self.data_types:
                            dict['flow_path']=os.path.join(self.data_base_path, flow_path)
                        if 'rgb' in self.data_types:
                            dict['rgb_path']=os.path.join(self.data_base_path, rgb_path)
                        if 'smartwatch' in self.data_types:
                            dict['smartwatch_path']=os.path.join(self.data_base_path, smartwatch_path)
                        dict['start_frame']=start_frame
                        dict['end_frame']=end_frame
                        dict['start_t']=step['start_t']
                        dict['end_t']=step['end_t']
                        dict['keystep_label']=label
                        dict['keystep_id']=keystep_id
                        dict['subject']=subject['subject_id']
                        dict['trial']=trial['trial_id']

                        self.data.append(dict)

    def _get_clips(self, data, frames_per_clip):
        # Function to split data into smaller clips
        clips = []
        num_clips = math.ceil(len(data) / frames_per_clip)
        for i in range(num_clips):
            start_idx = i * frames_per_clip
            end_idx = min((i + 1) * frames_per_clip, len(data))
            clips.append(data[start_idx:end_idx])
        return clips

    def _generate_clip_indices(self):
        # Generate indices that represent which clip (from which item) will be returned in __getitem__
        self.clip_indices = []
        for item_idx, item in enumerate(self.data):
            num_frames = item['end_frame'] - item['start_frame']
            num_clips = math.ceil(num_frames / self.frames_per_clip) if self.frames_per_clip else 1
            for clip_idx in range(num_clips):
                self.clip_indices.append((item_idx, clip_idx))

    def __len__(self):
        # The length should now be based on the number of clips, not items
        return len(self.clip_indices)
    def __getitem__(self, idx):
        # Get the actual item and the clip index for this sample
        item_idx, clip_idx = self.clip_indices[idx]
        item = self.data[item_idx]
        
        # Initialize variables
        frames = []
        flow = torch.zeros(0)
        rgb = torch.zeros(0)
        audio = torch.zeros(1, 0)
        sw_acc = torch.zeros(0)

        # Load video if available
        if 'video' in self.data_types:
            video_path = item['video_path']
            video_reader = VideoReader(video_path, "video")
            video_reader.seek(item['start_t'])
            for frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], video_reader):
                if frame['pts'] >= item['start_t']:
                    img_tensor = transform(frame['data'])
                    frames.append(img_tensor)
            frames = torch.stack(frames) if frames else torch.zeros(0)

        # Load flow if available
        if 'flow' in self.data_types:
            flow_path = item['flow_path']
            flow = torch.from_numpy(np.load(flow_path))[item['start_frame']:item['end_frame']]

        # Load rgb if available
        if 'rgb' in self.data_types:
            rgb_path = item['rgb_path']
            rgb = torch.from_numpy(np.load(rgb_path))[item['start_frame']:item['end_frame']]

        # Load audio if available
        if 'audio' in self.data_types and 'video' in self.data_types:  # Check if audio is available
            video_path = item['video_path']
            audio_reader = VideoReader(video_path, "audio")
            audio_reader.seek(item['start_t'])
            audio_clips = [audio_frame['data'] for audio_frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], audio_reader)]
            audio = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)

        # Load smartwatch data if available
        if 'smartwatch' in self.data_types:
            smartwatch_path = item['smartwatch_path']
            with open(smartwatch_path, 'r') as f:
                lines = f.readlines()[1:]
            sw_acc = [line.strip() for line in lines][item['start_frame']:item['end_frame']]
            # Process smartwatch data into tensors as before
            acc_x, acc_y, acc_z = [], [], []
            acc_x = [float(l.split(',')[0]) for l in sw_acc]
            acc_y = [float(l.split(',')[1]) for l in sw_acc]
            acc_z = [float(l.split(',')[2]) for l in sw_acc]
            sw_acc = torch.from_numpy(np.array([acc_x, acc_y, acc_z])).float()
            # permute to (batch, frames, channels)
            sw_acc = sw_acc.permute(1, 0)


        output = {
            'frames': frames,
            'flow': flow,
            'rgb': rgb,
            'audio': audio,
            'smartwatch': sw_acc,
            'keystep_label': item['keystep_label'],
            'keystep_id': item['keystep_id'],
            'start_frame': item['start_frame'],
            'end_frame': item['end_frame'],
            'start_t': item['start_t'],
            'end_t': item['end_t'],
            'subject_id': item['subject'],
            'trial_id': item['trial']
        }

        # Return the output
        return output

