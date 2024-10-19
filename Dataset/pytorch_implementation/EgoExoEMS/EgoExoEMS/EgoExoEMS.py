# import os
# import json
# import math
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision.io import VideoReader
# from torchvision import transforms
# from collections import OrderedDict
# import itertools

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
# ])

# def collate_fn(batch):
#     # Get max sequence length for each modality
#     max_video_len = max([clip['frames'].shape[0] for clip in batch])
#     max_audio_len = max([clip['audio'].shape[0] for clip in batch])
#     max_flow_len = max([clip['flow'].shape[0] for clip in batch])
#     max_rgb_len = max([clip['rgb'].shape[0] for clip in batch])

#     padded_clips = []
#     padded_audio_clips = []
#     padded_flow_clips = []
#     padded_rgb_clips = []
#     keystep_labels = []
#     keystep_ids = []
#     start_frames = []
#     end_frames = []
#     start_ts = []
#     end_ts = []
#     subject_ids = []
#     trial_ids = []

#     for b in batch:
#         video_clip = b['frames']
#         audio_clip = b['audio']
#         flow_clip = b['flow']
#         rgb_clip = b['rgb']
#         label = b['keystep_label']
#         keystep_id = b['keystep_id']
#         start_frame = b['start_frame']
#         end_frame = b['end_frame']
#         start_t = b['start_t']
#         end_t = b['end_t']
#         subject_id = b['subject_id']
#         trial_id = b['trial_id']

#         # Pad video frames
#         video_pad_size = max_video_len - video_clip.shape[0]
#         if video_pad_size > 0:
#             pad = torch.zeros((video_pad_size, *video_clip.shape[1:]))
#             video_clip = torch.cat([video_clip, pad], dim=0)

#         # Pad audio
#         audio_pad_size = max_audio_len - audio_clip.shape[0]
#         if audio_pad_size > 0:
#             audio_pad = torch.zeros((audio_pad_size, *audio_clip.shape[1:]))
#             audio_clip = torch.cat([audio_clip, audio_pad], dim=0)

#         # Pad flow data
#         flow_pad_size = max_flow_len - flow_clip.shape[0]
#         if flow_pad_size > 0:
#             flow_pad = torch.zeros((flow_pad_size, *flow_clip.shape[1:]))
#             flow_clip = torch.cat([flow_clip, flow_pad], dim=0)

#         # Pad rgb data
#         rgb_pad_size = max_rgb_len - rgb_clip.shape[0]
#         if rgb_pad_size > 0:
#             rgb_pad = torch.zeros((rgb_pad_size, *rgb_clip.shape[1:]))
#             rgb_clip = torch.cat([rgb_clip, rgb_pad], dim=0)

#         padded_clips.append(video_clip)
#         padded_audio_clips.append(audio_clip)
#         padded_flow_clips.append(flow_clip)
#         padded_rgb_clips.append(rgb_clip)
#         keystep_labels.append(label)
#         keystep_ids.append(keystep_id)
#         start_frames.append(start_frame)
#         end_frames.append(end_frame)
#         start_ts.append(start_t)
#         end_ts.append(end_t)
#         subject_ids.append(subject_id)
#         trial_ids.append(trial_id)

#     padded_clips = torch.stack(padded_clips)
#     padded_audio_clips = torch.stack(padded_audio_clips)
#     padded_flow_clips = torch.stack(padded_flow_clips)
#     padded_rgb_clips = torch.stack(padded_rgb_clips)
#     keystep_ids = torch.tensor(keystep_ids)
#     start_frames = torch.tensor(start_frames)
#     end_frames = torch.tensor(end_frames)
#     start_ts = torch.tensor(start_ts)
#     end_ts = torch.tensor(end_ts)

#     return {
#         'frames': padded_clips,
#         'audio': padded_audio_clips,
#         'flow': padded_flow_clips,
#         'rgb': padded_rgb_clips,
#         'keystep_label': keystep_labels,
#         'keystep_id': keystep_ids,
#         'start_frame': start_frames,
#         'end_frame': end_frames,
#         'start_t': start_ts,
#         'end_t': end_ts,
#         'subject_id': subject_ids,
#         'trial_id': trial_ids
#     }

# class EgoExoEMSDataset(Dataset):
#     def __init__(self, annotation_file, data_base_path, fps, 
#                 frames_per_clip=None, transform=None,
#                 data_types=['smartwatch'],
#                 split='train',
#                 audio_sample_rate=48000):
        
#         self.annotation_file = annotation_file
#         self.data_base_path = data_base_path
#         self.fps = fps
#         self.frames_per_clip = frames_per_clip  # Store frames_per_clip
#         self.transform = transform
#         self.audio_sample_rate = audio_sample_rate
#         self.data_types=data_types
#         self.split=split
#         self.data = []
#         self.clip_indices = []  # This will store (item_idx, clip_idx) tuples

        
#         self._load_annotations()
#         self._generate_clip_indices()

#     def _load_annotations(self):
#         with open(self.annotation_file, 'r') as f:
#             annotations = json.load(f)
        
#         for subject in annotations['subjects']:
#             for trial in subject['trials']:
#                 avail_streams = trial['streams']
                
#                 if 'video' not in self.data_types:
#                     video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
#                 if 'flow' in self.data_types:
#                     flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
#                 if 'rgb' in self.data_types:
#                     rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
#                 if 'smartwatch' in self.data_types:
#                     smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
#                     #read split file
#                     with open(os.path.join('Annotations/splits/cpr_quality/subject_splits.json'), 'r') as sw_file:
#                         splits = json.load(sw_file)
#                         if self.split=='train':
#                             subjects = splits['train'].split(',')
#                         elif self.split=='validation':
#                             subjects = splits['validation'].split(',')

#                 if video_path or flow_path or rgb_path or smartwatch_path:
#                     keysteps = trial['keysteps']
#                     for step in keysteps:
#                         start_frame = math.floor(step['start_t'] * self.fps)
#                         end_frame = math.ceil(step['end_t'] * self.fps)
#                         label = step['label']
#                         keystep_id = step['class_id']

#                         dict={}
#                         if 'video' in self.data_types:
#                             dict['video_path']=os.path.join(self.data_base_path, video_path)
#                         if 'flow' in self.data_types:
#                             dict['flow_path']=os.path.join(self.data_base_path, flow_path)
#                         if 'rgb' in self.data_types:
#                             dict['rgb_path']=os.path.join(self.data_base_path, rgb_path)
#                         if 'smartwatch' in self.data_types:
#                             dict['smartwatch_path']=os.path.join(self.data_base_path, smartwatch_path[1:])
#                             #skip if subject not in split
#                             if not subject['subject_id'] in subjects:
#                                 continue
#                         dict['start_frame']=start_frame
#                         dict['end_frame']=end_frame
#                         dict['start_t']=step['start_t']
#                         dict['end_t']=step['end_t']
#                         dict['keystep_label']=label
#                         dict['keystep_id']=keystep_id
#                         dict['subject']=subject['subject_id']
#                         dict['trial']=trial['trial_id']

#                         self.data.append(dict)

#     def _get_clips(self, data, frames_per_clip):
#         # Function to split data into smaller clips
#         clips = []
#         num_clips = math.ceil(len(data) / frames_per_clip)
#         for i in range(num_clips):
#             start_idx = i * frames_per_clip
#             end_idx = min((i + 1) * frames_per_clip, len(data))
#             clips.append(data[start_idx:end_idx])
#         return clips

#     def _generate_clip_indices(self):
#         # Generate indices that represent which clip (from which item) will be returned in __getitem__
#         self.clip_indices = []
#         for item_idx, item in enumerate(self.data):
#             num_frames = item['end_frame'] - item['start_frame']
#             num_clips = math.ceil(num_frames / self.frames_per_clip) if self.frames_per_clip else 1
#             for clip_idx in range(num_clips):
#                 self.clip_indices.append((item_idx, clip_idx))

#     def __len__(self):
#         # The length should now be based on the number of clips, not items
#         return len(self.clip_indices)

#     def __getitem__(self, idx):
#         # Get the actual item and the clip index for this sample
#         item_idx, clip_idx = self.clip_indices[idx]
#         item = self.data[item_idx]
#         if 'video' in self.data_types:
#             video_path = item['video_path']
#         if 'flow' in self.data_types:
#             flow_path = item['flow_path']
#         if 'rgb' in self.data_types:
#             rgb_path = item['rgb_path']
#         if 'smartwatch' in self.data_types:
#             smartwatch_path = item['smartwatch_path']

#         start_frame = item['start_frame']
#         end_frame = item['end_frame']
#         start_t = item['start_t']
#         end_t = item['end_t']
#         keystep_label = item['keystep_label']
#         keystep_id = item['keystep_id']
#         subject_id = item['subject']
#         trial_id = item['trial']
        

#         # Load video
#         frames = []
#         if 'video' in self.data_types:
#             video_reader = VideoReader(video_path, "video")
#             for frame in itertools.takewhile(lambda x: x['pts'] <= end_t, video_reader.seek(start_t)):
#                 img_tensor = transform(frame['data'])
#                 frames.append(img_tensor)
#             frames = torch.stack(frames)

#         # Load flow data
#         flow=[]
#         if 'flow' in self.data_types:
#             flow = torch.from_numpy(np.load(flow_path))
#             flow = flow[start_frame:end_frame]

#         # Load rgb data
#         rgb=[]
#         if 'rgb' in self.data_types:
#             rgb = torch.from_numpy(np.load(rgb_path))
#             rgb = rgb[start_frame:end_frame]

#         # Calculate audio sample range based on frames_per_clip
#         audio=[]
#         if 'audio' in self.data_types:
#             if self.frames_per_clip:
#                 clip_duration = self.frames_per_clip / self.fps  # Time in seconds for frames_per_clip
#                 audio_samples_per_clip = int(clip_duration * self.audio_sample_rate)

#                 audio_reader = VideoReader(video_path, "audio")
#                 audio_clips = []
#                 for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
#                     audio_clips.append(audio_frame['data'])
#                 audio_clips = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)
#                 # Get the slice corresponding to the current clip
#                 audio = audio_clips[clip_idx * audio_samples_per_clip : (clip_idx + 1) * audio_samples_per_clip]
#             else:
#                 audio_reader = VideoReader(video_path, "audio")
#                 audio = []
#                 for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
#                     audio.append(audio_frame['data'])
#                 audio = torch.cat(audio, dim=0) if audio else torch.zeros(1, 0)

#         if 'smartwatch' in self.data_types:
#             with open(smartwatch_path, 'r') as f:
#                 lines = f.readlines()
#                 sw_acc = [l.strip() for l in lines][1:]
            
#         # Split into smaller clips if frames_per_clip is specified
#         if self.frames_per_clip:
#             # Return the specific clip from the item based on clip_idx
#             if 'video' in self.data_types:
#                 frames_clips = self._get_clips(frames, self.frames_per_clip)
#                 frames = frames_clips[clip_idx]
#             if 'flow' in self.data_types:
#                 flow_clips = self._get_clips(flow, self.frames_per_clip)
#                 flow = flow_clips[clip_idx]
#             if 'rgb' in self.data_types:
#                 rgb_clips = self._get_clips(rgb, self.frames_per_clip)
#                 rgb = rgb_clips[clip_idx]
#             if 'smartwatch' in self.data_types:
#                 sw_acc_clips = self._get_clips(sw_acc, self.frames_per_clip)
#                 sw_acc = sw_acc_clips[clip_idx]
#                 lines_left=self.frames_per_clip-len(sw_acc)
#                 #pad the clips to be the same length
#                 if lines_left>0:
#                     sw_acc.append(['0,0,0']*lines_left)
#                 acc_x, acc_y, acc_z = [], [], []
#                 acc_x=[float(l.split(',')[0]) for l in sw_acc]
#                 acc_y=[float(l.split(',')[1]) for l in sw_acc]
#                 acc_z=[float(l.split(',')[2]) for l in sw_acc]
#                 sw_acc=np.array([acc_x, acc_y, acc_z])

    
#         output = {
#             'frames': frames,
#             'audio': audio,
#             'flow': flow,
#             'rgb': rgb,
#             'smartwatch': sw_acc,
#             'keystep_label': keystep_label,
#             'keystep_id': keystep_id,
#             'start_frame': start_frame,
#             'end_frame': end_frame,
#             'start_t': start_t,
#             'end_t': end_t,
#             'subject_id': subject_id,
#             'trial_id': trial_id
#         }

#         return output




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
    max_smartwatch_len = max([clip['smartwatch'].shape[1] for clip in batch if isinstance(clip.get('smartwatch', None), torch.Tensor)], default=0)
    max_depth_sensor_len = max([clip['depth_sensor'].shape[1] for clip in batch if isinstance(clip.get('depth_sensor', None), torch.Tensor)], default=0)

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

        # Pad smartwatch data if available
        if 'smartwatch' in b and isinstance(b['smartwatch'], torch.Tensor):
            smartwatch_clip = b['smartwatch']
            smartwatch_pad_size = max_smartwatch_len - smartwatch_clip.shape[1]
            if smartwatch_pad_size > 0:
                smartwatch_pad = torch.zeros((smartwatch_clip.shape[0], smartwatch_pad_size))
                smartwatch_clip = torch.cat([smartwatch_clip, smartwatch_pad], dim=1)
            padded_smartwatch_clips.append(smartwatch_clip)

        # Pad depth_sensor data if available
        if 'depth_sensor' in b and isinstance(b['depth_sensor'], torch.Tensor):
            depth_sensor_clip = b['depth_sensor']
            depth_sensor_pad_size = max_depth_sensor_len - depth_sensor_clip.shape[1]
            if depth_sensor_pad_size > 0:
                depth_sensor_pad = torch.zeros((depth_sensor_clip.shape[0], depth_sensor_pad_size))
                depth_sensor_clip = torch.cat([depth_sensor_clip, depth_sensor_pad], dim=1)
            padded_depth_sensor_clips.append(depth_sensor_clip)

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
    output['smartwatch'] = torch.stack(padded_smartwatch_clips) if padded_smartwatch_clips else torch.zeros(0)
    output['depth_sensor'] = torch.stack(padded_depth_sensor_clips) if padded_depth_sensor_clips else torch.zeros(0)

    return output


class EgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, data_base_path, fps, 
                frames_per_clip=None, transform=None,
                data_types=['smartwatch'],
                split='train',
                activity='chest_compressions',
                split_path=None,
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
        self.split = split
        self.activity = activity
        
        self._load_splits(split_path)
        self._load_annotations()
        self._generate_clip_indices()

    def _load_splits(self,split_path):
        if split_path:
            with open(split_path, 'r') as f:
                splits = json.load(f)
                self.split_subjects = splits[self.split].split(',')
    
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
                depth_sensor_path = None

                # Check for each data type and retrieve the corresponding file path
                if 'video' in self.data_types:
                    video_path = avail_streams.get('egocam_rgb_audio', {}).get('file_path', None)
                if 'flow' in self.data_types:
                    flow_path = avail_streams.get('i3d_flow', {}).get('file_path', None)
                if 'rgb' in self.data_types:
                    rgb_path = avail_streams.get('i3d_rgb', {}).get('file_path', None)
                if 'smartwatch' in self.data_types:
                    smartwatch_path = avail_streams.get('smartwatch_imu', {}).get('file_path', None)
                if 'depth_sensor' in self.data_types:
                    depth_sensor_path = avail_streams.get('vl6180_ToF_depth', {}).get('file_path', None)

                # Skip the trial if any required data type is not available
                if ('video' in self.data_types and not video_path) or \
                ('flow' in self.data_types and not flow_path) or \
                ('rgb' in self.data_types and not rgb_path) or \
                ('smartwatch' in self.data_types and not smartwatch_path) or \
                ('depth_sensor' in self.data_types and not depth_sensor_path):
                    print(f"[Warning] Skipping trial {trial['trial_id']} for subject {subject['subject_id']} due to missing data")
                    continue

                if video_path or flow_path or rgb_path or smartwatch_path or depth_sensor_path:
                    keysteps = trial['keysteps']
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.floor(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        #filter out subjects not in split
                        if subject['subject_id'] not in self.split_subjects:
                            continue
                        #filter out activities
                        if label != self.activity:
                            continue

                        data_dict = {}
                        if 'video' in self.data_types:
                            data_dict['video_path'] = os.path.join(self.data_base_path, video_path)
                        if 'flow' in self.data_types:
                            data_dict['flow_path'] = os.path.join(self.data_base_path, flow_path)
                        if 'rgb' in self.data_types:
                            data_dict['rgb_path'] = os.path.join(self.data_base_path, rgb_path)
                        if 'smartwatch' in self.data_types:
                            data_dict['smartwatch_path'] = os.path.join(self.data_base_path, smartwatch_path[1:])
                        if 'depth_sensor' in self.data_types:
                            data_dict['depth_sensor_path'] = os.path.join(self.data_base_path, depth_sensor_path[1:])
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

        # # Load audio if available
        # if 'audio' in self.data_types and 'video' in self.data_types:  # Check if audio is available
        #     video_path = item['video_path']
        #     audio_reader = VideoReader(video_path, "audio")
        #     audio_clips = [audio_frame['data'] for audio_frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], audio_reader)]
        #     audio = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)

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
            if self.frames_per_clip:
                clip_duration = self.frames_per_clip / self.fps  # Time in seconds for frames_per_clip
                audio_samples_per_clip = int(clip_duration * self.audio_sample_rate)
                audio_reader = VideoReader(video_path, "audio")
                audio_clips = []
                for audio_frame in itertools.takewhile(lambda x: x['pts'] <= item['end_t'], audio_reader.seek(item['start_t'])):
                    audio_clips.append(audio_frame['data'])
                audio_clips = torch.cat(audio_clips, dim=0) if audio_clips else torch.zeros(1, 0)
                # Get the slice corresponding to the current clip
                audio = audio_clips[clip_idx * audio_samples_per_clip : (clip_idx + 1) * audio_samples_per_clip]
            else:
                audio_reader = VideoReader(video_path, "audio")
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
            frames.shape[0] if frames is torch.is_tensor(frames) else float('inf'),
            flow.shape[0] if flow is not None and flow.shape[0] > 0 else float('inf'),
            rgb.shape[0] if rgb is not None and rgb.shape[0] > 0 else float('inf'),
            sw_acc.shape[0] if sw_acc is not None and sw_acc.shape[0] > 0 else float('inf'),
            depth_sensor_readings.shape[0] if depth_sensor_readings is not None and depth_sensor_readings.shape[0] > 0 else float('inf')
        )

        # Make sure min_length is a valid integer
        if min_length == float('inf'):
            min_length = 0  # Handle the case where none of the data types are available

        # Truncate all modalities to the minimum length
        if frames is torch.is_tensor(frames) and frames.shape[0] > 0:
            frames = frames[:min_length]
        if flow is not None and flow.shape[0] > 0:
            flow = flow[:min_length]
        if rgb is not None and rgb.shape[0] > 0:
            rgb = rgb[:min_length]
        if sw_acc is not None and sw_acc.shape[0] > 0:
            sw_acc = sw_acc[:min_length]
        if depth_sensor_readings is not None and depth_sensor_readings.shape[0] > 0:
            depth_sensor_readings = depth_sensor_readings[:min_length]


        output = {
            'frames': frames,
            'flow': flow,
            'rgb': rgb,
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

