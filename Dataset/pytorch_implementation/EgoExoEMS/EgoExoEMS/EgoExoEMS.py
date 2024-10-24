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
    padded_resnet_clips = []
    padded_resnet_exo_clips = []  # Added for resnet_exo modality
    padded_smartwatch_clips = []
    padded_depth_sensor_clips = []
    keystep_labels = []
    keystep_ids = []
    start_frames = []
    end_frames = []
    start_ts = []
    end_ts = []
    subject_ids = []
    trial_ids = []

    # Initialize max lengths for each modality, check if each is available
    max_video_len = max([clip['frames'].shape[0] for clip in batch if isinstance(clip.get('frames', None), torch.Tensor)], default=0)
    max_audio_len = max([clip['audio'].shape[0] for clip in batch if isinstance(clip.get('audio', None), torch.Tensor)], default=0)
    max_flow_len = max([clip['flow'].shape[0] for clip in batch if isinstance(clip.get('flow', None), torch.Tensor)], default=0)
    max_rgb_len = max([clip['rgb'].shape[0] for clip in batch if isinstance(clip.get('rgb', None), torch.Tensor)], default=0)
    max_resnet_len = max([clip['resnet'].shape[0] for clip in batch if isinstance(clip.get('resnet', None), torch.Tensor)], default=0)
    max_resnet_exo_len = max([clip['resnet_exo'].shape[0] for clip in batch if isinstance(clip.get('resnet_exo', None), torch.Tensor)], default=0)  # Added for resnet_exo modality
    max_smartwatch_len = max([clip['smartwatch'].shape[0] for clip in batch if isinstance(clip.get('smartwatch', None), torch.Tensor)], default=0)
    max_depth_sensor_len = max([clip['depth_sensor'].shape[0] for clip in batch if isinstance(clip.get('depth_sensor', None), torch.Tensor)], default=0)

    # print(batch)
    
    for b in batch:
        # Pad video frames if available
        if 'frames' in b and isinstance(b['frames'], torch.Tensor):
            video_clip = b['frames']
            video_pad_size = max_video_len - video_clip.shape[0]
            if video_pad_size > 0:
                video_pad = torch.zeros((video_pad_size, *video_clip.shape[1:]))
                video_clip = torch.cat([video_clip, video_pad], dim=0)
            padded_clips.append(video_clip)

        # Pad audio if available
        if 'audio' in b and isinstance(b['audio'], torch.Tensor):
            audio_clip = b['audio']
            audio_pad_size = max_audio_len - audio_clip.shape[0]
            if audio_pad_size > 0:
                audio_pad = torch.zeros((audio_pad_size, *audio_clip.shape[1:]))
                audio_clip = torch.cat([audio_clip, audio_pad], dim=0)
            padded_audio_clips.append(audio_clip)

        # Pad flow data if available
        if 'flow' in b and isinstance(b['flow'], torch.Tensor):
            flow_clip = b['flow']
            flow_pad_size = max_flow_len - flow_clip.shape[0]
            if flow_pad_size > 0:
                flow_pad = torch.zeros((flow_pad_size, *flow_clip.shape[1:]))
                flow_clip = torch.cat([flow_clip, flow_pad], dim=0)
            padded_flow_clips.append(flow_clip)

        # Pad rgb data if available
        if 'rgb' in b and isinstance(b['rgb'], torch.Tensor):
            rgb_clip = b['rgb']
            rgb_pad_size = max_rgb_len - rgb_clip.shape[0]
            if rgb_pad_size > 0:
                rgb_pad = torch.zeros((rgb_pad_size, *rgb_clip.shape[1:]))
                rgb_clip = torch.cat([rgb_clip, rgb_pad], dim=0)
            padded_rgb_clips.append(rgb_clip)

        # Pad resnet data if available
        if 'resnet' in b and isinstance(b['resnet'], torch.Tensor):
            resnet_clip = b['resnet']
            resnet_pad_size = max_resnet_len - resnet_clip.shape[0]
            if resnet_pad_size > 0:
                resnet_pad = torch.zeros((resnet_pad_size, *resnet_clip.shape[1:]))
                resnet_clip = torch.cat([resnet_clip, resnet_pad], dim=0)
            padded_resnet_clips.append(resnet_clip)

        # Pad resnet_exo data if available
        if 'resnet_exo' in b and isinstance(b['resnet_exo'], torch.Tensor):
            resnet_exo_clip = b['resnet_exo']
            resnet_exo_pad_size = max_resnet_exo_len - resnet_exo_clip.shape[0]
            if resnet_exo_pad_size > 0:
                resnet_exo_pad = torch.zeros((resnet_exo_pad_size, *resnet_exo_clip.shape[1:]))
                resnet_exo_clip = torch.cat([resnet_exo_clip, resnet_exo_pad], dim=0)
            padded_resnet_exo_clips.append(resnet_exo_clip)

        # Pad smartwatch data if available
        if 'smartwatch' in b and isinstance(b['smartwatch'], torch.Tensor):
            smartwatch_clip = b['smartwatch']
            if max_smartwatch_len > 0:
                smartwatch_pad_size = max_smartwatch_len - smartwatch_clip.shape[0]
                if smartwatch_pad_size > 0:
                    smartwatch_pad = torch.zeros((smartwatch_pad_size, smartwatch_clip.shape[1]))
                    smartwatch_clip = torch.cat([smartwatch_clip, smartwatch_pad], dim=0)
            padded_smartwatch_clips.append(smartwatch_clip)

        # Pad depth_sensor data if available
        if 'depth_sensor' in b and isinstance(b['depth_sensor'], torch.Tensor):
            depth_sensor_clip = b['depth_sensor']
            if max_depth_sensor_len > 0:
                depth_sensor_pad_size = max_depth_sensor_len - depth_sensor_clip.shape[0]
                if depth_sensor_pad_size > 0:
                    depth_sensor_pad = torch.zeros((depth_sensor_pad_size, depth_sensor_clip.shape[1]))
                    depth_sensor_clip = torch.cat([depth_sensor_clip, depth_sensor_pad], dim=0)
            padded_depth_sensor_clips.append(depth_sensor_clip)

        # max_length = max(max_rgb_len, max_resnet_len, max_resnet_exo_len, max_smartwatch_len, max_depth_sensor_len, max_video_len,  max_flow_len)
        # pad_length = max_length - len(b['keystep_id']) 
        
        # if pad_length > 0:
        #     # repeat last element of keystep_id and keystep_label
        #     b['keystep_id'] = b['keystep_id'] + [b['keystep_id'][-1]]*pad_length
        #     b['start_frame'] = b['start_frame'] + [b['start_frame'][-1]]*pad_length
        #     b['end_frame'] = b['end_frame'] + [b['end_frame'][-1]]*pad_length
        #     b['start_t'] = b['start_t'] + [b['start_t'][-1]]*pad_length
        #     b['end_t'] = b['end_t'] + [b['end_t'][-1]]*pad_length
            
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
    output['frames'] = torch.stack(padded_clips) if padded_clips else torch.zeros(0)
    output['audio'] = torch.stack(padded_audio_clips) if padded_audio_clips else torch.zeros(0)
    output['flow'] = torch.stack(padded_flow_clips) if padded_flow_clips else torch.zeros(0)
    output['rgb'] = torch.stack(padded_rgb_clips) if padded_rgb_clips else torch.zeros(0)
    output['resnet'] = torch.stack(padded_resnet_clips) if padded_resnet_clips else torch.zeros(0)
    output['resnet_exo'] = torch.stack(padded_resnet_exo_clips) if padded_resnet_exo_clips else torch.zeros(0)  # Added for resnet_exo
    output['smartwatch'] = torch.stack(padded_smartwatch_clips) if padded_smartwatch_clips else torch.zeros(0)
    output['depth_sensor'] = torch.stack(padded_depth_sensor_clips) if padded_depth_sensor_clips else torch.zeros(0)

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
        self.data_types = data_types
        
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
                audio_path = None
                flow_path = None
                rgb_path = None
                resnet_path = None
                resnet_exo_path = None  # Added for resnet_exo modality
                smartwatch_path = None
                depth_sensor_path = None

                # Check for each data type and retrieve the corresponding file path
                if 'video' in self.data_types:
                    video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'audio' in self.data_types:
                    audio_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'flow' in self.data_types:
                    flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                if 'rgb' in self.data_types:
                    rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                if 'resnet' in self.data_types:
                    resnet_path = avail_streams.get('resnet50', {}).get('file_path', None)
                if 'resnet_exo' in self.data_types:
                    resnet_exo_path = avail_streams.get('resnet50-exo', {}).get('file_path', None)  # Adjust key as needed
                if 'smartwatch' in self.data_types:
                    smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
                if 'depth_sensor' in self.data_types:
                    depth_sensor_path = avail_streams.get('vl6180_ToF_depth', {}).get('file_path', None)

                # Skip the trial if any required data type is not available
                if ('video' in self.data_types and not video_path) or \
                ('audio' in self.data_types and not audio_path) or \
                ('flow' in self.data_types and not flow_path) or \
                ('rgb' in self.data_types and not rgb_path) or \
                ('resnet' in self.data_types and not resnet_path) or \
                ('resnet_exo' in self.data_types and not resnet_exo_path) or \
                ('smartwatch' in self.data_types and not smartwatch_path) or \
                ('depth_sensor' in self.data_types and not depth_sensor_path):
                    print(f"[Warning] Skipping trial {trial['trial_id']} for subject {subject['subject_id']} due to missing data")
                    continue

                if video_path or audio_path or flow_path or rgb_path or resnet_path or resnet_exo_path or smartwatch_path or depth_sensor_path:
                    keysteps = trial['keysteps']
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.floor(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        data_dict = {}
                        if 'video' in self.data_types:
                            data_dict['video_path'] = os.path.join(self.data_base_path, video_path)
                        if 'audio' in self.data_types:
                            data_dict['audio_path'] = os.path.join(self.data_base_path, audio_path)
                        if 'flow' in self.data_types:
                            data_dict['flow_path'] = os.path.join(self.data_base_path, flow_path)
                        if 'rgb' in self.data_types:
                            data_dict['rgb_path'] = os.path.join(self.data_base_path, rgb_path)
                        if 'resnet' in self.data_types:
                            data_dict['resnet_path'] = os.path.join(self.data_base_path, resnet_path)
                        if 'resnet_exo' in self.data_types:
                            data_dict['resnet_exo_path'] = os.path.join(self.data_base_path, resnet_exo_path)
                        if 'smartwatch' in self.data_types:
                            data_dict['smartwatch_path'] = os.path.join(self.data_base_path, smartwatch_path)
                        if 'depth_sensor' in self.data_types:
                            data_dict['depth_sensor_path'] = os.path.join(self.data_base_path, depth_sensor_path)
                        data_dict['start_frame'] = start_frame
                        data_dict['end_frame'] = end_frame
                        data_dict['start_t'] = step['start_t']
                        data_dict['end_t'] = step['end_t']
                        data_dict['keystep_label'] = label
                        data_dict['keystep_id'] = keystep_id
                        data_dict['subject'] = subject['subject_id']
                        data_dict['trial'] = trial['trial_id']

                        self.data.append(data_dict)

    def _get_clips(self, data, frames_per_clip):
        # Ensure that data is not empty and frames_per_clip is greater than 0
        if len(data) == 0 or frames_per_clip <= 0:
            return []
        
        clips = []
        num_clips = math.ceil(len(data) / frames_per_clip)
        
        for i in range(num_clips):
            start_idx = i * frames_per_clip
            end_idx = min((i + 1) * frames_per_clip, len(data))
            clip = data[start_idx:end_idx]
            # Ensure that the clip has data before appending
            clips.append(clip)
        
        return clips
    
    def _generate_clip_indices(self):
        # Clear clip_indices to start fresh
        self.clip_indices = []
        
        for item_idx, item in enumerate(self.data):
            num_frames = item['end_frame'] - item['start_frame']
            
            # Skip the item if it doesn't have enough frames
            if num_frames <= 0:
                continue
            
            num_clips = math.floor(num_frames / self.frames_per_clip) if self.frames_per_clip else 1
            if num_clips == 0:
                num_clips = 1
            for clip_idx in range(num_clips):
                # Ensure clip_idx is valid
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
        resnet = torch.zeros(0)
        resnet_exo = torch.zeros(0)  # Added for resnet_exo modality
        audio = torch.zeros(1, 0)
        sw_acc = torch.zeros(0)
        depth_sensor_readings = torch.zeros(0)

        
        # Load video if available
        if 'video' in self.data_types:
            video_path = item['video_path']
            video_reader = VideoReader(video_path, "video")
            video_reader.seek(item['start_t'])
            for frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], video_reader):
                if frame['pts'] > item['start_t']:
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

        # Load resnet if available
        if 'resnet' in self.data_types:
            resnet_path = item['resnet_path']
            resnet = torch.from_numpy(np.load(resnet_path))[item['start_frame']:item['end_frame']]

        # Load resnet_exo if available
        if 'resnet_exo' in self.data_types:
            resnet_exo_path = item['resnet_exo_path']
            resnet_exo = torch.from_numpy(np.load(resnet_exo_path))[item['start_frame']:item['end_frame']]

        # Load smartwatch data if available
        if 'smartwatch' in self.data_types:
            smartwatch_path = item['smartwatch_path']
            with open(smartwatch_path, 'r') as f:
                lines = f.readlines()[1:]
            sw_acc = [line.strip() for line in lines][item['start_frame']:item['end_frame']]
            acc_x = [float(l.split(',')[0]) for l in sw_acc]
            acc_y = [float(l.split(',')[1]) for l in sw_acc]
            acc_z = [float(l.split(',')[2]) for l in sw_acc]
            sw_acc = torch.from_numpy(np.array([acc_x, acc_y, acc_z])).float()
            sw_acc = sw_acc.permute(1, 0)  # (frames, channels)

        # Load depth_sensor data if available
        if 'depth_sensor' in self.data_types:
            depth_sensor_path = item['depth_sensor_path']
            with open(depth_sensor_path, 'r') as f:
                lines = f.readlines()[1:]
            depth_sensor_readings = [line.strip() for line in lines][item['start_frame']:item['end_frame']]
            depth_reading = [float(l.split(',')[0]) for l in depth_sensor_readings]
            depth_sensor_readings = torch.from_numpy(np.array([depth_reading])).float()
            depth_sensor_readings = depth_sensor_readings.permute(1, 0)

        if 'audio' in self.data_types:  # Check if audio is available
            audio_path = item['audio_path']

            if self.frames_per_clip:
                clip_duration = self.frames_per_clip / self.fps  # Time in seconds for frames_per_clip
                audio_samples_per_clip = int(clip_duration * self.audio_sample_rate)
                audio_reader = VideoReader(audio_path, "audio")
                audio_clips = []
                for audio_frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], audio_reader.seek(item['start_t'])):
                    audio_clips.append(audio_frame['data'])
                audio_clips = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)
                # Get the slice corresponding to the current clip
                audio = audio_clips[clip_idx * audio_samples_per_clip : (clip_idx + 1) * audio_samples_per_clip]
            else:
                audio_reader = VideoReader(audio_path, "audio")
                audio = []
                for audio_frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], audio_reader.seek(item['start_t'])):
                    audio.append(audio_frame['data'])
                audio = torch.cat(audio, dim=0) if audio else torch.zeros(1, 0)


        # Split into smaller clips if frames_per_clip is specified
        if self.frames_per_clip:
            if 'video' in self.data_types:
                frames_clips = self._get_clips(frames, self.frames_per_clip)
                frames = frames_clips[clip_idx]
                # Pad if less than frames_per_clip
                if frames.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - frames.shape[0]
                    frames = torch.cat([frames, torch.zeros((pad_size, *frames.shape[1:]))], dim=0)


                clip_duration = self.frames_per_clip / self.fps


            if 'flow' in self.data_types:
                flow_clips = self._get_clips(flow, self.frames_per_clip)
                clip_idx = min(clip_idx, len(flow_clips) - 1)
                flow = flow_clips[clip_idx]
                # Pad if less than frames_per_clip
                if flow.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - flow.shape[0]
                    flow = torch.cat([flow, torch.zeros((pad_size, *flow.shape[1:]))], dim=0)

            if 'rgb' in self.data_types:
                rgb_clips = self._get_clips(rgb, self.frames_per_clip)
                clip_idx = min(clip_idx, len(rgb_clips) - 1)
                rgb = rgb_clips[clip_idx]
                # Pad if less than frames_per_clip
                if rgb.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - rgb.shape[0]
                    rgb = torch.cat([rgb, torch.zeros((pad_size, *rgb.shape[1:]))], dim=0)

            if 'resnet' in self.data_types:
                resnet_clips = self._get_clips(resnet, self.frames_per_clip)
                clip_idx = min(clip_idx, len(resnet_clips) - 1)
                resnet = resnet_clips[clip_idx]
                # Pad if less than frames_per_clip
                if resnet.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - resnet.shape[0]
                    resnet = torch.cat([resnet, torch.zeros((pad_size, *resnet.shape[1:]))], dim=0)

            if 'resnet_exo' in self.data_types:
                resnet_exo_clips = self._get_clips(resnet_exo, self.frames_per_clip)
                clip_idx = min(clip_idx, len(resnet_exo_clips) - 1)
                resnet_exo = resnet_exo_clips[clip_idx]
                # Pad if less than frames_per_clip
                if resnet_exo.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - resnet_exo.shape[0]
                    resnet_exo = torch.cat([resnet_exo, torch.zeros((pad_size, *resnet_exo.shape[1:]))], dim=0)

            if 'smartwatch' in self.data_types:
                sw_acc_clips = self._get_clips(sw_acc, self.frames_per_clip)
                sw_acc = sw_acc_clips[clip_idx]
                # Pad if less than frames_per_clip
                if sw_acc.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - sw_acc.shape[0]
                    sw_acc = torch.cat([sw_acc, torch.zeros((pad_size, sw_acc.shape[1]))], dim=0)

            if 'depth_sensor' in self.data_types:
                depth_sensor_clips = self._get_clips(depth_sensor_readings, self.frames_per_clip)
                depth_sensor_readings = depth_sensor_clips[clip_idx]
                # Pad if less than frames_per_clip
                if depth_sensor_readings.shape[0] < self.frames_per_clip:
                    pad_size = self.frames_per_clip - depth_sensor_readings.shape[0]
                    depth_sensor_readings = torch.cat([depth_sensor_readings, torch.zeros((pad_size, depth_sensor_readings.shape[1]))], dim=0)

        # Calculate the minimum length of all available modalities
        min_length = min(
            frames.shape[0] if isinstance(frames, torch.Tensor) and frames.shape[0] > 0 else float('inf'),
            flow.shape[0] if isinstance(flow, torch.Tensor) and flow.shape[0] > 0 else float('inf'),
            rgb.shape[0] if isinstance(rgb, torch.Tensor) and rgb.shape[0] > 0 else float('inf'),
            resnet.shape[0] if isinstance(resnet, torch.Tensor) and resnet.shape[0] > 0 else float('inf'),
            resnet_exo.shape[0] if isinstance(resnet_exo, torch.Tensor) and resnet_exo.shape[0] > 0 else float('inf'),
            sw_acc.shape[0] if isinstance(sw_acc, torch.Tensor) and sw_acc.shape[0] > 0 else float('inf'),
            depth_sensor_readings.shape[0] if isinstance(depth_sensor_readings, torch.Tensor) and depth_sensor_readings.shape[0] > 0 else float('inf')
        )

        # Make sure min_length is a valid integer
        if min_length == float('inf'):
            min_length = 0  # Handle the case where none of the data types are available

        # Truncate all modalities to the minimum length
        if isinstance(frames, torch.Tensor) and frames.shape[0] > 0:
            frames = frames[:min_length]
        if isinstance(flow, torch.Tensor) and flow.shape[0] > 0:
            flow = flow[:min_length]
        if isinstance(rgb, torch.Tensor) and rgb.shape[0] > 0:
            rgb = rgb[:min_length]
        if isinstance(resnet, torch.Tensor) and resnet.shape[0] > 0:
            resnet = resnet[:min_length]
        if isinstance(resnet_exo, torch.Tensor) and resnet_exo.shape[0] > 0:
            resnet_exo = resnet_exo[:min_length]
        if isinstance(sw_acc, torch.Tensor) and sw_acc.shape[0] > 0:
            sw_acc = sw_acc[:min_length]
        if isinstance(depth_sensor_readings, torch.Tensor) and depth_sensor_readings.shape[0] > 0:
            depth_sensor_readings = depth_sensor_readings[:min_length]


        output = {
            'frames': frames,
            'flow': flow,
            'rgb': rgb,
            'resnet': resnet,
            'resnet_exo': resnet_exo,  # Added for resnet_exo modality
            'audio': audio,
            'smartwatch': sw_acc,
            'depth_sensor': depth_sensor_readings,
            'keystep_label': item['keystep_label'],
            'keystep_id': item['keystep_id'],
            'start_frame': item['start_frame'],
            'end_frame': item['end_frame'],
            'start_t': item['start_t'],
            'end_t': item['end_t'],
            'subject_id': item['subject'],
            'trial_id': item['trial']
        }

        return output




############################################## Tumbling Window ##############################################

def window_collate_fn(batch, frames_per_clip=30):
    # Initialize empty lists for all possible modalities
    padded_video_clips = []
    padded_audio_clips = []
    padded_flow_clips = []
    padded_rgb_clips = []
    padded_resnet_clips = []
    padded_resnet_exo_clips = []  # Added for resnet_exo modality
    padded_smartwatch_clips = []
    padded_depth_sensor_clips = []
    keystep_labels = []
    keystep_ids = []
    window_start_frames = []
    window_end_frames = []
    start_frames = []
    end_frames = []
    start_ts = []
    end_ts = []
    subject_ids = []
    trial_ids = []

    # Initialize max lengths for each modality, check if each is available
    max_flow_len = max([clip['flow'].shape[0] for clip in batch if isinstance(clip.get('flow', None), torch.Tensor)], default=0)
    max_audio_len = max([clip['audio'].shape[0] for clip in batch if isinstance(clip.get('audio', None), torch.Tensor)], default=0)
    max_rgb_len = max([clip['rgb'].shape[0] for clip in batch if isinstance(clip.get('rgb', None), torch.Tensor)], default=0)
    max_resnet_len = max([clip['resnet'].shape[0] for clip in batch if isinstance(clip.get('resnet', None), torch.Tensor)], default=0)
    max_resnet_exo_len = max([clip['resnet_exo'].shape[0] for clip in batch if isinstance(clip.get('resnet_exo', None), torch.Tensor)], default=0)  # Added for resnet_exo modality
    max_smartwatch_len = max([clip['smartwatch'].shape[0] for clip in batch if isinstance(clip.get('smartwatch', None), torch.Tensor)], default=0)
    max_depth_sensor_len = max([clip['depth_sensor'].shape[0] for clip in batch if isinstance(clip.get('depth_sensor', None), torch.Tensor)], default=0)

    # print("max_flow_len:",max_flow_len)
    # print("max_audio_len:",max_audio_len)
    # print("max_rgb_len:",max_rgb_len)
    # print("max_resnet_len:",max_resnet_len)
    # print("max_resnet_exo_len:",max_resnet_exo_len)
    # print("max_smartwatch_len:",max_smartwatch_len)
    # print("max_depth_sensor_len:",max_depth_sensor_len)
    
    pad_length = 0
    for b in batch:
        # print("b:",b)
        if 'video' in b and isinstance(b['video'], torch.Tensor):
            video_clip = b['video']
            video_pad_size = frames_per_clip - video_clip.shape[0]
            if video_pad_size > 0:
                video_pad = torch.zeros((video_pad_size, *video_clip.shape[1:]))
                video_clip = torch.cat([video_clip, video_pad], dim=0)
            padded_video_clips.append(video_clip)
        
        # Pad audio if available
        if 'audio' in b and isinstance(b['audio'], torch.Tensor):
            audio_clip = b['audio']
            audio_pad_size = max_audio_len - audio_clip.shape[0]
            # print("audio_pad_size:",audio_pad_size)
            if audio_pad_size > 0:
                audio_pad = torch.zeros((audio_pad_size, *audio_clip.shape[1:]))
                # print("audio_pad:",audio_pad.shape)
                audio_clip = torch.cat([audio_clip, audio_pad], dim=0)
            padded_audio_clips.append(audio_clip)


        # Pad flow data if available
        if 'flow' in b and isinstance(b['flow'], torch.Tensor):
            flow_clip = b['flow']
            flow_pad_size = frames_per_clip - flow_clip.shape[0]
            if flow_pad_size > 0:
                flow_pad = torch.zeros((flow_pad_size, *flow_clip.shape[1:]))
                flow_clip = torch.cat([flow_clip, flow_pad], dim=0)
            padded_flow_clips.append(flow_clip)

        # Pad rgb data if available
        if 'rgb' in b and isinstance(b['rgb'], torch.Tensor):
            rgb_clip = b['rgb']
            rgb_pad_size = frames_per_clip - rgb_clip.shape[0]
            if rgb_pad_size > 0:
                rgb_pad = torch.zeros((rgb_pad_size, *rgb_clip.shape[1:]))
                rgb_clip = torch.cat([rgb_clip, rgb_pad], dim=0)
            padded_rgb_clips.append(rgb_clip)

        # Pad resnet data if available
        if 'resnet' in b and isinstance(b['resnet'], torch.Tensor):
            resnet_clip = b['resnet']
            resnet_pad_size = frames_per_clip - resnet_clip.shape[0]
            if resnet_pad_size > 0:
                resnet_pad = torch.zeros((resnet_pad_size, *resnet_clip.shape[1:]))
                resnet_clip = torch.cat([resnet_clip, resnet_pad], dim=0)
            padded_resnet_clips.append(resnet_clip)

        # Pad resnet_exo data if available
        if 'resnet_exo' in b and isinstance(b['resnet_exo'], torch.Tensor):
            resnet_exo_clip = b['resnet_exo']
            resnet_exo_pad_size = frames_per_clip - resnet_exo_clip.shape[0]
            if resnet_exo_pad_size > 0:
                resnet_exo_pad = torch.zeros((resnet_exo_pad_size, *resnet_exo_clip.shape[1:]))
                resnet_exo_clip = torch.cat([resnet_exo_clip, resnet_exo_pad], dim=0)
            padded_resnet_exo_clips.append(resnet_exo_clip)

        # Pad smartwatch data if available
        if 'smartwatch' in b and isinstance(b['smartwatch'], torch.Tensor):
            smartwatch_clip = b['smartwatch']
            # print("smartwatch_clip:",smartwatch_clip.shape)
            if max_smartwatch_len > 0:
                smartwatch_pad_size = frames_per_clip - smartwatch_clip.shape[0]
                # print("smartwatch_pad_size:",smartwatch_pad_size)
                if smartwatch_pad_size > 0:
                    smartwatch_pad = torch.zeros((smartwatch_pad_size, smartwatch_clip.shape[1]))
                    # print("padding_smartwatch", smartwatch_pad.shape)
                    smartwatch_clip = torch.cat([smartwatch_clip, smartwatch_pad], dim=0)
            padded_smartwatch_clips.append(smartwatch_clip)

        # Pad depth_sensor data if available
        if 'depth_sensor' in b and isinstance(b['depth_sensor'], torch.Tensor):
            depth_sensor_clip = b['depth_sensor']
            # print("depth_sensor_clip:",depth_sensor_clip.shape)
            if max_depth_sensor_len > 0:
                depth_sensor_pad_size = frames_per_clip - depth_sensor_clip.shape[0]
                # print("depth_sensor_pad_size:",depth_sensor_pad_size)
                if depth_sensor_pad_size > 0:
                    depth_sensor_pad = torch.zeros((depth_sensor_pad_size, depth_sensor_clip.shape[1]))
                    # print("padding_depth_sensor", depth_sensor_pad.shape)
                    depth_sensor_clip = torch.cat([depth_sensor_clip, depth_sensor_pad], dim=0)
            padded_depth_sensor_clips.append(depth_sensor_clip)



        pad_length = frames_per_clip - len(b['keystep_id']) 
        if pad_length > 0:
            # repeat last element of keystep_id and keystep_label
            b['keystep_id'] = b['keystep_id'] + [b['keystep_id'][-1]]*pad_length
            b['start_frame'] = b['start_frame'] + [b['start_frame'][-1]]*pad_length
            b['end_frame'] = b['end_frame'] + [b['end_frame'][-1]]*pad_length
            b['start_t'] = b['start_t'] + [b['start_t'][-1]]*pad_length
            b['end_t'] = b['end_t'] + [b['end_t'][-1]]*pad_length
        
        # Collect other fields
        keystep_labels.append(b['keystep_label'])
        keystep_ids.append(b['keystep_id'])
        start_frames.append(b['start_frame'])
        end_frames.append(b['end_frame'])
        start_ts.append(b['start_t'])
        end_ts.append(b['end_t'])
        subject_ids.append(b['subject_id'])
        trial_ids.append(b['trial_id'])
        window_start_frames.append(b['window_start_frame'])
        window_end_frames.append(b['window_end_frame'])

    output = {
        'keystep_label': keystep_labels,
        'keystep_id': torch.tensor(keystep_ids),
        'start_frame': torch.tensor(start_frames),
        'end_frame': torch.tensor(end_frames),
        'start_t': torch.tensor(start_ts),
        'end_t': torch.tensor(end_ts),
        'subject_id': subject_ids,
        'trial_id': trial_ids,
        'window_start_frame': torch.tensor(window_start_frames),
        'window_end_frame': torch.tensor(window_end_frames)
    }

    # Only include modality data if it exists
    output['video'] = torch.stack(padded_video_clips) if padded_video_clips else torch.zeros(0)
    output['audio'] = torch.stack(padded_audio_clips) if padded_audio_clips else torch.zeros(0)
    output['flow'] = torch.stack(padded_flow_clips) if padded_flow_clips else torch.zeros(0)
    output['rgb'] = torch.stack(padded_rgb_clips) if padded_rgb_clips else torch.zeros(0)
    output['resnet'] = torch.stack(padded_resnet_clips) if padded_resnet_clips else torch.zeros(0)
    output['resnet_exo'] = torch.stack(padded_resnet_exo_clips) if padded_resnet_exo_clips else torch.zeros(0)  # Added for resnet_exo
    output['smartwatch'] = torch.stack(padded_smartwatch_clips) if padded_smartwatch_clips else torch.zeros(0)
    output['depth_sensor'] = torch.stack(padded_depth_sensor_clips) if padded_depth_sensor_clips else torch.zeros(0)

    return output


class WindowEgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, data_base_path, fps, 
                frames_per_clip=30, transform=None,
                data_types=['resnet'], audio_sample_rate=48000):
        
        self.annotation_file = annotation_file
        self.data_base_path = data_base_path
        self.fps = fps
        self.frames_per_clip = frames_per_clip  # Store frames_per_clip
        self.transform = transform
        self.data = []
        self.clip_indices = []  # This will store (item_idx, clip_idx) tuples
        self.data_types = data_types
        self.audio_sample_rate = audio_sample_rate
        
        self.data_dict = None
        self._load_annotations()
        self._split_windows()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        subject_dict = {}
        for subject in annotations['subjects']:
            trial_dict ={}
            for trial in subject['trials']:
                avail_streams = trial['streams']
                
                # Initialize paths to None by default
                video_path = None
                audio_path = None
                flow_path = None
                rgb_path = None
                resnet_path = None
                resnet_exo_path = None  # Added for resnet_exo modality
                smartwatch_path = None
                depth_sensor_path = None

                # Check for each data type and retrieve the corresponding file path
                if 'video' in self.data_types:
                    video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'audio' in self.data_types:
                    audio_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'flow' in self.data_types:
                    flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                if 'rgb' in self.data_types:
                    rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                if 'resnet' in self.data_types:
                    resnet_path = avail_streams.get('resnet50', {}).get('file_path', None)
                if 'resnet_exo' in self.data_types:
                    resnet_exo_path = avail_streams.get('resnet50-exo', {}).get('file_path', None)  # Adjust key as needed
                if 'smartwatch' in self.data_types:
                    smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
                if 'depth_sensor' in self.data_types:
                    depth_sensor_path = avail_streams.get('vl6180_ToF_depth', {}).get('file_path', None)

                # Skip the trial if any required data type is not available
                if ('flow' in self.data_types and not flow_path) or \
                ('audio' in self.data_types and not audio_path) or \
                ('video' in self.data_types and not video_path) or \
                ('rgb' in self.data_types and not rgb_path) or \
                ('resnet' in self.data_types and not resnet_path) or \
                ('resnet_exo' in self.data_types and not resnet_exo_path) or \
                ('smartwatch' in self.data_types and not smartwatch_path) or \
                ('depth_sensor' in self.data_types and not depth_sensor_path):
                    print(f"[Warning] Skipping trial {trial['trial_id']} for subject {subject['subject_id']} due to missing data")
                    continue
                
                if video_path or audio_path or flow_path or rgb_path or resnet_path or resnet_exo_path or smartwatch_path or depth_sensor_path:
                    keysteps = trial['keysteps']
                    keysteps_dict = []
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.floor(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        data_dict = {}
                        if 'video' in self.data_types:
                            data_dict['video_path'] = os.path.join(self.data_base_path, video_path)
                        if 'audio' in self.data_types:
                            data_dict['audio_path'] = os.path.join(self.data_base_path, audio_path)
                        if 'flow' in self.data_types:
                            data_dict['flow_path'] = os.path.join(self.data_base_path, flow_path)
                        if 'rgb' in self.data_types:
                            data_dict['rgb_path'] = os.path.join(self.data_base_path, rgb_path)
                        if 'resnet' in self.data_types:
                            data_dict['resnet_path'] = os.path.join(self.data_base_path, resnet_path)
                        if 'resnet_exo' in self.data_types:
                            data_dict['resnet_exo_path'] = os.path.join(self.data_base_path, resnet_exo_path)
                        if 'smartwatch' in self.data_types:
                            data_dict['smartwatch_path'] = os.path.join(self.data_base_path, smartwatch_path)
                        if 'depth_sensor' in self.data_types:
                            data_dict['depth_sensor_path'] = os.path.join(self.data_base_path, depth_sensor_path)
                        data_dict['start_frame'] = start_frame
                        data_dict['end_frame'] = end_frame
                        data_dict['start_t'] = step['start_t']
                        data_dict['end_t'] = step['end_t']
                        data_dict['keystep_label'] = label
                        data_dict['keystep_id'] = keystep_id
                        data_dict['subject'] = subject['subject_id']
                        data_dict['trial'] = trial['trial_id']

                        keysteps_dict.append(data_dict)
                        self.data.append(data_dict)

                        trial_dict[trial['trial_id']] = keysteps_dict
                subject_dict[subject['subject_id']] = trial_dict

        self.data_dict = subject_dict

    def _split_windows(self):
        print("Splitting data to windows")
        windowed_clips = []
        current_window = []  # Initialize an empty current window
        accumulated_frames = 0  # Track how many frames have been accumulated in the current window

        current_subject = None
        current_trial = None


        for i, item in enumerate(self.data):
            start_frame = item['start_frame']
            end_frame = item['end_frame']
            keystep_id = item['keystep_id']
            keystep_label = item['keystep_label']
            subject_id = item['subject']
            trial_id = item['trial']
            num_frames_total = end_frame - start_frame

                
        # If the subject or trial changes, store the current window and reset it
            if subject_id != current_subject or trial_id != current_trial:
                if len(current_window) > 0:
                    windowed_clips.append(current_window)
                current_window = []  # Reset the current window
                accumulated_frames = 0  # Reset the frame counter
                current_subject = subject_id  # Update current subject
                current_trial = trial_id  # Update current trial

            # Loop through frames from the start to the end of the current keystep
            for j in range(start_frame, end_frame):
                frame_data = {}
                frame_data['frame'] = j

                # Copy the relevant data for the current frame
                for key, value in item.items():
                    if isinstance(value, (int, float, str)):
                        frame_data[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        if len(value) == num_frames_total:
                            frame_data[key] = value[j - start_frame]
                    else:
                        frame_data[key] = value

                # Append the current frame data to the window
                current_window.append(frame_data)
                accumulated_frames += 1


                # Once we reach the window size, store the window and reset
                if accumulated_frames == self.frames_per_clip:
                    windowed_clips.append(current_window)
                    current_window = []  # Reset the current window
                    accumulated_frames = 0  # Reset the frame counter

        # # dump the data to a file
        # with open('data.json', 'w') as f:
        #     json.dump(windowed_clips, f)
            
        self.data = windowed_clips
        print(f"Total windowed clips: {len(windowed_clips)}")

            

    def __len__(self):
        # The length should now be based on the number of clips, not items
        # total_clips = len(self.data) // self.frames_per_clip if self.frames_per_clip else len(self.data)
        # dump the data to a file
        # with open('data.json', 'w') as f:
        #     json.dump(self.data, f)
        total_clips = len(self.data)
        return total_clips

    def __getitem__(self, idx):
        window = self.data[idx]
        # print("window", window)

        first_frame_of_clip = window[0]['frame']
        last_frame_of_clip = window[-1]['frame']+1
        
        # print(f"getting clip {idx} with frame {first_frame_of_clip} to {last_frame_of_clip} of length ({last_frame_of_clip - first_frame_of_clip})")

        # Initialize dictionaries to hold accumulated frames for each modality
        batch_video = []
        batch_audio = []
        batch_flow = []
        batch_rgb = []
        batch_resnet = []
        batch_resnet_exo = []
        batch_sw_acc = []
        batch_depth_sensor = []

        # Initialize lists for metadata
        keystep_labels = []
        keystep_ids = []
        start_frames = []
        end_frames = []
        start_ts = []
        end_ts = []
        subject_ids = []
        trial_ids = []


        # Initialize variables
        frames = []
        flow = torch.zeros(0)
        rgb = torch.zeros(0)
        resnet = torch.zeros(0)
        resnet_exo = torch.zeros(0)  # Added for resnet_exo modality
        sw_acc = torch.zeros(0)
        depth_sensor_readings = torch.zeros(0)

        if 'video' in self.data_types:
            video_path = window[0]['video_path']
            clip_start_t = first_frame_of_clip / self.fps
            clip_end_t = last_frame_of_clip / self.fps

            print(f"loading video from {clip_start_t} to {clip_end_t} from file {video_path}")

            video_reader = VideoReader(video_path, "video")
            video_reader.seek(clip_start_t)
            for frame in itertools.takewhile(lambda x: x['pts'] <= clip_end_t, video_reader):
                if frame['pts'] > clip_start_t:
                    img_tensor = transform(frame['data'])
                    frames.append(img_tensor)
            frames = torch.stack(frames) if frames else torch.zeros(0)
            batch_video.append(frames)       
                               
        if 'audio' in self.data_types:
            audio_path = window[0]['audio_path']
            clip_start_t = first_frame_of_clip / self.fps
            clip_end_t = last_frame_of_clip / self.fps

            # print(f"loading audio from {clip_start_t} to {clip_end_t} from file {audio_path}")

            audio_reader = VideoReader(audio_path, "audio")
            audio_clips = []
            for audio_frame in itertools.takewhile(lambda x: x['pts'] <= clip_end_t, audio_reader.seek(clip_start_t)):
                audio_clips.append(audio_frame['data'])
            audio_clips = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)
            batch_audio.append(audio_clips)


        if 'flow' in self.data_types:
            flow_path = window[0]['flow_path']
            flow_npy = np.load(flow_path)
            flow_length = len(flow_npy)
            flow = torch.from_numpy(np.load(flow_path))[first_frame_of_clip:last_frame_of_clip]
                # pad the flow tensor to the i
            batch_flow.append(flow)

        # Load rgb if available
        if 'rgb' in self.data_types:
            rgb_path =  window[0]['rgb_path']
            rgb_npy = np.load(rgb_path)
            rgb_length = len(rgb_npy)
            rgb = torch.from_numpy(np.load(rgb_path))[first_frame_of_clip:last_frame_of_clip]
                # pad the rgb tensor to the i
            batch_rgb.append(rgb)

        # Load resnet if available
        if 'resnet' in self.data_types:
            resnet_path =  window[0]['resnet_path']
            resnet_npy = np.load(resnet_path)
            resnet_length = len(resnet_npy)
            resnet = torch.from_numpy(np.load(resnet_path))[first_frame_of_clip:last_frame_of_clip]
            batch_resnet.append(resnet)

        # Load resnet_exo if available
        if 'resnet_exo' in self.data_types:
            resnet_exo_path =  window[0]['resnet_exo_path']
            resnet_exo = torch.from_numpy(np.load(resnet_exo_path))[first_frame_of_clip:last_frame_of_clip]
            batch_resnet_exo.append(resnet_exo)

        # Load smartwatch data if available
        if 'smartwatch' in self.data_types:
            smartwatch_path =  window[0]['smartwatch_path']
            with open(smartwatch_path, 'r') as f:
                lines = f.readlines()[1:]
            sw_acc = [line.strip() for line in lines][first_frame_of_clip:last_frame_of_clip]
            acc_x = [float(l.split(',')[0]) for l in sw_acc]
            acc_y = [float(l.split(',')[1]) for l in sw_acc]
            acc_z = [float(l.split(',')[2]) for l in sw_acc]
            sw_acc = torch.from_numpy(np.array([acc_x, acc_y, acc_z])).float()
            sw_acc = sw_acc.permute(1, 0)  # (frames, channels)
            batch_sw_acc.append(sw_acc)

        # Load depth_sensor data if available
        if 'depth_sensor' in self.data_types:
            depth_sensor_path =  window[0]['depth_sensor_path']
            with open(depth_sensor_path, 'r') as f:
                lines = f.readlines()[1:]
            depth_sensor_readings = [line.strip() for line in lines][first_frame_of_clip:last_frame_of_clip]
            depth_reading = [float(l.split(',')[0]) for l in depth_sensor_readings]
            depth_sensor_readings = torch.from_numpy(np.array([depth_reading])).float()
            depth_sensor_readings = depth_sensor_readings.permute(1, 0)
            batch_depth_sensor.append(depth_sensor_readings)

        for frame in window:
            # Accumulate metadata for each frame
            keystep_labels.append( frame['keystep_label'])
            keystep_ids.append( frame['keystep_id'])
            start_frames.append( frame['start_frame'])
            end_frames.append( frame['end_frame'])
            start_ts.append( frame['start_t'])
            end_ts.append( frame['end_t'])
            subject_ids.append( frame['subject'])
            trial_ids.append( frame['trial'])

        first_frame_of_clip = torch.tensor(first_frame_of_clip)
        last_frame_of_clip = torch.tensor(last_frame_of_clip)
        # print("batch_resnet:",batch_resnet[0].shape)
        # Stack frames to form a batch of frames_per_clip
        output = {
            'video': (batch_video[0]) if batch_video else torch.zeros(1,0),
            'audio': (batch_audio[0]) if batch_audio else torch.zeros(1,0),
            'flow': (batch_flow[0]) if batch_flow else torch.zeros(0),
            'rgb': (batch_rgb[0]) if batch_rgb else torch.zeros(0),
            'resnet': (batch_resnet[0]) if batch_resnet else torch.zeros(0),
            'resnet_exo': (batch_resnet_exo[0]) if batch_resnet_exo else torch.zeros(0),
            'smartwatch': (batch_sw_acc[0]) if batch_sw_acc else torch.zeros(0),
            'depth_sensor': (batch_depth_sensor[0]) if batch_depth_sensor else torch.zeros(0),

            # Metadata (individual per frame)
            'keystep_label': keystep_labels,
            'keystep_id': keystep_ids,
            'start_frame': start_frames,
            'end_frame': end_frames,
            'window_start_frame': first_frame_of_clip,
            'window_end_frame': last_frame_of_clip,
            'start_t': start_ts,
            'end_t': end_ts,
            'subject_id': subject_ids,
            'trial_id': trial_ids
        }

        return output


