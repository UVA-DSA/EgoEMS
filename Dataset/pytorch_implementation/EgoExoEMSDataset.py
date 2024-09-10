import os
import json
import math
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision import transforms

#load itertools
import itertools

transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def collate_fn(batch):
    max_len = max([clip['frames'].shape[0] for clip in batch])

    padded_clips = []
    padded_audio_clips = []
    keystep_labels = []
    keystep_ids = []
    start_frames = []
    end_frames = []
    subject_ids = []
    trial_ids = []

    for b in batch:
        clip = b['frames']
        audio_clip = b['audio']
        label = b['keystep_label']
        keystep_id = b['keystep_id']
        start_frame = b['start_frame']
        end_frame = b['end_frame']
        subject_id = b['subject_id']
        trial_id = b['trial_id']

        pad_size = max_len - clip.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, *clip.shape[1:]))
            clip = torch.cat([clip, pad], dim=0)
            
            # Pad the audio clip as well with zeros
            audio_pad = torch.zeros((pad_size, *audio_clip.shape[1:]))
            audio_clip = torch.cat([audio_clip, audio_pad], dim=0)
            
        padded_clips.append(clip)
        padded_audio_clips.append(audio_clip)
        keystep_labels.append(label)
        keystep_ids.append(keystep_id)
        start_frames.append(start_frame)
        end_frames.append(end_frame)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)

    padded_clips = torch.stack(padded_clips)
    padded_audio_clips = torch.stack(padded_audio_clips)
    keystep_ids = torch.tensor(keystep_ids)
    start_frames = torch.tensor(start_frames)
    end_frames = torch.tensor(end_frames)

    return {
        'frames': padded_clips,
        'audio': padded_audio_clips,
        'keystep_label': keystep_labels,
        'keystep_id': keystep_ids,
        'start_frame': start_frames,
        'end_frame': end_frames,
        'subject_id': subject_ids,
        'trial_id': trial_ids
    }

class EgoExoEMSDataset(Dataset):
    def __init__(self, annotation_file, video_base_path, fps, frames_per_clip=None, transform=None, audio_sample_rate=16000):
        self.annotation_file = annotation_file
        self.video_base_path = video_base_path
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
                stream = trial['streams'].get('egocam_rgb_audio', None)
                if stream:
                    video_path = os.path.join(self.video_base_path, stream['file_path'])
                    keysteps = trial['keysteps']
                    
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.ceil(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']
                        
                        print(step['start_t'], step['end_t'])
                        print(self.fps)
                        print(start_frame, end_frame)
                        print(step['label'])

                        if self.frames_per_clip is None:
                            self.data.append({
                                'video_path': video_path,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'start_t': step['start_t'],
                                'end_t': step['end_t'],
                                'keystep_label': label,
                                'keystep_id': keystep_id,
                                'subject': subject['subject_id'],
                                'trial': trial['trial_id'] 
                            })
                        else:
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
        start_t = item['start_t']
        end_t = item['end_t']
        keystep_label = item['keystep_label']
        keystep_id = item['keystep_id']
        subject_id = item['subject']
        trial_id = item['trial']
        
        print("Loading video:", video_path)
        print("Start frame:", start_frame)
        print("End frame:", end_frame)
        print("Start time:", start_t)
        print("End time:", end_t)
        print("Keystep label:", keystep_label)
        print("Keystep ID:", keystep_id)
        print("Subject ID:", subject_id)
        print("Trial ID:", trial_id)
        
              
        # Load video and audio using torchvision's VideoReader
        video_reader = VideoReader(video_path, "video")
        audio_reader = VideoReader(video_path, "audio")

        # Extract frames for the specific segment
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_t, video_reader.seek(start_t)):
            img_tensor = transform(frame['data'])
            frames.append(img_tensor)
        
        frames = torch.stack(frames)

        # Extract corresponding audio for the same segment
        audio_reader.seek(start_frame / self.fps)
        audio = []
        for audio_frame in itertools.takewhile(lambda x: x['pts'] <= end_t, audio_reader.seek(start_t)):
            audio.append(audio_frame['data'])
            
        audio = torch.cat(audio, dim=0) if audio else torch.zeros(1, 0)  # Handle empty case

        output = {
            'frames': frames,
            'audio': audio,
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

if __name__ == '__main__':
    dataset = EgoExoEMSDataset(annotation_file='../../Annotations/main_annotation.json',
                               video_base_path='',
                               fps=30, 
                               transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    for batch in data_loader:
        print(batch.keys())
        print(batch['frames'].shape, batch['audio'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'], batch['subject_id'], batch['trial_id'])
        break
