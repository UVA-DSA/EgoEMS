import os
import json
import math
import cv2
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
                if stream and 'keysteps' in stream:
                    video_path = os.path.join(self.video_base_path, stream['file_path'])
                    keysteps = stream['keysteps']
                    
                    for step in keysteps:
                        start_frame = math.floor(step['start_t'] * self.fps)
                        end_frame = math.ceil(step['end_t'] * self.fps)
                        label = step['label']
                        keystep_id = step['class_id']

                        if self.frames_per_clip is None:
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

        frames = torch.stack(frames)  # Shape will be [T, C, H, W]

        # Extract the corresponding audio clip
        audio_start_time = start_frame / self.fps
        audio_end_time = end_frame / self.fps
        command = f'ffmpeg -i "{video_path}" -ss {audio_start_time} -to {audio_end_time} -ar {self.audio_sample_rate} -f wav -'
        waveform, sample_rate = torchaudio.load(command)

        # Resample if necessary
        if sample_rate != self.audio_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.audio_sample_rate)
            waveform = resampler(waveform)

        output = {
            'frames': frames,
            'audio': waveform,
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
                               fps=30, 
                               transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    for batch in data_loader:
        print(batch.keys())
        print(batch['frames'].shape, batch['audio'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'], batch['subject_id'], batch['trial_id'])
        break
