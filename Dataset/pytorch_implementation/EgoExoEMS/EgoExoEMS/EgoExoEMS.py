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
    def __init__(self, annotation_file, data_base_path, fps, 
                frames_per_clip=None, transform=None,
                data_types=['smartwatch'],
                split='train',
                audio_sample_rate=48000):
        
        self.annotation_file = annotation_file
        self.data_base_path = data_base_path
        self.fps = fps
        self.frames_per_clip = frames_per_clip  # Store frames_per_clip
        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.data_types=data_types
        self.split=split
        self.data = []
        self.clip_indices = []  # This will store (item_idx, clip_idx) tuples

        
        self._load_annotations()
        self._generate_clip_indices()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for subject in annotations['subjects']:
            for trial in subject['trials']:
                avail_streams = trial['streams']
                
                if 'video' not in self.data_types:
                    video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'flow' in self.data_types:
                    flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                if 'rgb' in self.data_types:
                    rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                if 'smartwatch' in self.data_types:
                    smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
                    #read split file
                    with open(os.path.join('Annotations/splits/cpr_quality/subject_splits.json'), 'r') as sw_file:
                        splits = json.load(sw_file)
                        if self.split=='train':
                            subjects = splits['train'].split(',')
                        elif self.split=='validation':
                            subjects = splits['validation'].split(',')

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
                            dict['smartwatch_path']=os.path.join(self.data_base_path, smartwatch_path[1:])
                            #skip if subject not in split
                            if not subject['subject_id'] in subjects:
                                continue
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
        if 'video' in self.data_types:
            video_path = item['video_path']
        if 'flow' in self.data_types:
            flow_path = item['flow_path']
        if 'rgb' in self.data_types:
            rgb_path = item['rgb_path']
        if 'smartwatch' in self.data_types:
            smartwatch_path = item['smartwatch_path']

        start_frame = item['start_frame']
        end_frame = item['end_frame']
        start_t = item['start_t']
        end_t = item['end_t']
        keystep_label = item['keystep_label']
        keystep_id = item['keystep_id']
        subject_id = item['subject']
        trial_id = item['trial']
        

        # Load video
        frames = []
        if 'video' in self.data_types:
            video_reader = VideoReader(video_path, "video")
            for frame in itertools.takewhile(lambda x: x['pts'] <= end_t, video_reader.seek(start_t)):
                img_tensor = transform(frame['data'])
                frames.append(img_tensor)
            frames = torch.stack(frames)

        # Load flow data
        flow=[]
        if 'flow' in self.data_types:
            flow = torch.from_numpy(np.load(flow_path))
            flow = flow[start_frame:end_frame]

        # Load rgb data
        rgb=[]
        if 'rgb' in self.data_types:
            rgb = torch.from_numpy(np.load(rgb_path))
            rgb = rgb[start_frame:end_frame]

        # Calculate audio sample range based on frames_per_clip
        audio=[]
        if 'audio' in self.data_types:
            if self.frames_per_clip:
                clip_duration = self.frames_per_clip / self.fps  # Time in seconds for frames_per_clip
                audio_samples_per_clip = int(clip_duration * self.audio_sample_rate)

                audio_reader = VideoReader(video_path, "audio")
                audio_clips = []
                for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
                    audio_clips.append(audio_frame['data'])
                audio_clips = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)
                # Get the slice corresponding to the current clip
                audio = audio_clips[clip_idx * audio_samples_per_clip : (clip_idx + 1) * audio_samples_per_clip]
            else:
                audio_reader = VideoReader(video_path, "audio")
                audio = []
                for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
                    audio.append(audio_frame['data'])
                audio = torch.cat(audio, dim=0) if audio else torch.zeros(1, 0)

        if 'smartwatch' in self.data_types:
            with open(smartwatch_path, 'r') as f:
                lines = f.readlines()
                sw_acc = [l.strip() for l in lines][1:]
            
        # Split into smaller clips if frames_per_clip is specified
        if self.frames_per_clip:
            # Return the specific clip from the item based on clip_idx
            if 'video' in self.data_types:
                frames_clips = self._get_clips(frames, self.frames_per_clip)
                frames = frames_clips[clip_idx]
            if 'flow' in self.data_types:
                flow_clips = self._get_clips(flow, self.frames_per_clip)
                flow = flow_clips[clip_idx]
            if 'rgb' in self.data_types:
                rgb_clips = self._get_clips(rgb, self.frames_per_clip)
                rgb = rgb_clips[clip_idx]
            if 'smartwatch' in self.data_types:
                sw_acc_clips = self._get_clips(sw_acc, self.frames_per_clip)
                sw_acc = sw_acc_clips[clip_idx]
                lines_left=self.frames_per_clip-len(sw_acc)
                #pad the clips to be the same length
                if lines_left>0:
                    sw_acc.append(['0,0,0']*lines_left)
                acc_x, acc_y, acc_z = [], [], []
                acc_x=[float(l.split(',')[0]) for l in sw_acc]
                acc_y=[float(l.split(',')[1]) for l in sw_acc]
                acc_z=[float(l.split(',')[2]) for l in sw_acc]
                sw_acc=np.array([acc_x, acc_y, acc_z])

    
        output = {
            'frames': frames,
            'audio': audio,
            'flow': flow,
            'rgb': rgb,
            'smartwatch': sw_acc,
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

