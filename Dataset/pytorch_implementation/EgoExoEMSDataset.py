import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import av
import numpy as np

class EgoExoEMSDataset(Dataset):
    def __init__(self, json_path, streams=None, labels='keysteps'):
        """
        Args:
            json_path (str): Path to the JSON file containing the data structure.
            streams (list): List of streams to include (e.g., ['exocam_rgbd', 'vl6180_ToF_depth']).
            labels (str): Either 'interventions' or 'keysteps' to specify which labels to load.
        """
        self.json_path = json_path
        self.streams = streams if streams is not None else []
        self.labels = labels
        self.data = self._load_data()

    def _load_data(self):
        # Load the JSON file
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)
        
        # Prepare the data structure
        data = []
        for subject in json_data['subjects']:
            for trial in subject['trials']:
                trial_data = {'subject_id': subject['subject_id'], 'trial_id': trial['trial_id']}
                for stream in self.streams:
                    if stream in trial['streams']:
                        stream_data = trial['streams'][stream]
                        trial_data[stream] = stream_data['file_path']
                        
                        # Extract labels (either 'interventions' or 'keysteps')
                        trial_data[f"{stream}_{self.labels}"] = stream_data.get(self.labels, [])
                
                data.append(trial_data)
        
        return data

    def _load_video(self, file_path):
        """
        Loads video from the given file path using torchvision's read_video.
        Args:
            file_path (str): Path to the video file.

        Returns:
            frames (Tensor): Video frames as a Tensor.
            frame_rate (float): Frame rate of the video.
        """
        print("File path: ", file_path) 
        container = av.open(file_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        # Calculate the frame step to match the target FPS
        frame_step = max(1, int(stream.average_rate / self.target_fps))

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_step == 0:
                img = frame.to_image()
                frames.append(torch.tensor(np.array(img)).permute(2, 0, 1))  # Convert to Tensor

        return frames, stream.average_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trial_data = self.data[idx]
        streams_data = {}
        labels_data = []

        for stream in self.streams:
            if stream in trial_data:
                # Load the video data
                video_path = trial_data[stream]
                video_frames, fps = self._load_video(video_path)
                streams_data[stream] = {
                    'frames': video_frames,
                    'fps': fps
                }
                
                # Load the labels
                stream_labels = trial_data.get(f"{stream}_{self.labels}", [])
                for label_info in stream_labels:
                    labels_data.append({
                        'stream': stream,
                        'start': label_info['start_t'],
                        'end': label_info['end_t'],
                        'label': label_info['label']
                    })

        return {
            'subject_id': trial_data['subject_id'],
            'trial_id': trial_data['trial_id'],
            'streams': streams_data,
            'labels': labels_data
        }


# path to json annotation
json_path = '../../Tools/output_structure.json'

streams = ['egocam_rgb_audio'] # 'vl6180_ToF_depth'
dataset = EgoExoEMSDataset(json_path, streams=streams, labels='keysteps')

# Access a sample
sample = dataset[0]
print(sample)
print(sample['streams']['egocam_rgb_audio']['frames'].shape)  # Shape of the video tensor

