import numpy as np
import matplotlib.pyplot as plt

#import from models folder transtcn
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from EgoExoEMS.EgoExoEMS import  WindowEgoExoEMSDataset, EgoExoEMSDataset, collate_fn, transform, window_collate_fn
from functools import partial
import torch.nn.functional as F
import numpy as np

import numpy as np
from collections import Counter

from torch.utils.data import DataLoader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def get_data_stats(data_loader):
    import torch
    depth_norm_vals=torch.empty(0)
    acc_vals=torch.empty(0)

    for i, batch in enumerate(data_loader):
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0
        depth_gt_min = torch.where(depth_gt_mask, depth_gt, torch.tensor(float('inf')))
        mins=depth_gt_min.min(dim=1).values
        depth_norm=depth_gt-mins.unsqueeze(1)
        depth_norm_vals=torch.concat([depth_norm_vals,depth_norm[depth_gt_mask]])
        acc_vals=torch.concat([acc_vals,data])

    print(f'max depth: {depth_norm_vals.max()}')
    print(f'min depth: {depth_norm_vals.min()}')
    print(f'max acc: {torch.max(torch.max(acc_vals,dim=0).values,dim=0).values}')
    print(f'min acc: {torch.min(torch.min(acc_vals,dim=0).values,dim=0).values}')


def moving_normalize(signal, window_size):
    # Initialize the normalized signal with zeros
    normalized_signal = np.zeros(signal.shape)
    
    # Calculate the half window size for indexing
    half_window = window_size // 2
    
    for i in range(len(signal)):
        # Determine the start and end of the window
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(signal))
        
        # Calculate local mean and standard deviation
        local_mean = np.mean(signal[start:end])
        local_std = np.std(signal[start:end])
        
        # Normalize the current value
        if local_std > 0:  # Avoid division by zero
            normalized_signal[i] = (signal[i] - local_mean) / local_std
        else:
            normalized_signal[i] = signal[i] - local_mean
    
    return normalized_signal


def plot_line(data):
    plt.plot(data)
    plt.show(block=True)


def find_peaks_and_valleys(signal, distance=10,height=0.2,prominence=(None, None),plot=False):
    from scipy.signal import find_peaks
    peaks, p_properties  = find_peaks(signal, distance=distance,height=height,prominence=prominence)
    valleys, v_properties = find_peaks(-signal, distance=distance,height=height,prominence=prominence)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(signal)
        plt.scatter(peaks, signal[peaks], c='r', label='Peaks')
        plt.scatter(valleys, signal[valleys], c='g', label='Valleys')
        plt.legend()
        plt.show(block=True)
    best_idx=-1
    if len(peaks)>1:
        best_idx=np.argmax([p_properties['prominences'].mean(),v_properties['prominences'].mean()])
    return peaks, valleys,best_idx

def detect_peaks_and_valleys_depth_sensor(depth_vals,mul=1,show=True):
    depth_vals_norm_=moving_normalize(depth_vals, 19)
    num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/len(depth_vals_norm_)
    dist=int(1/num_zero_crossings*mul)
    depth_vals_norm=moving_normalize(depth_vals, 60)
    GT_peaks,GT_valleys,idx=find_peaks_and_valleys(depth_vals_norm,distance=dist,height=0.2,plot=show)
    return GT_peaks,GT_valleys





# return train,val,test dataloaders using the EgoExoEMSDataset class
def eee_get_dataloaders(args):
    
    if(args.dataloader_params["task"] == 'classification'):
        print("*" * 10, "=" * 10, "*" * 10)
        print("Loading dataloader for Classification task")

        train_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

        val_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

        test_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])


        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)
        # print number of keys in the dictionary
        print("Train Number of classes: ", len(train_class_stats.keys()))

        val_class_stats = val_dataset._get_class_stats()
        print("val class stats: ", val_class_stats)
        # print number of keys in the dictionary
        print("Val Number of classes: ", len(val_class_stats.keys()))

        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))
    
    elif (args.dataloader_params["task"] == 'segmentation'):
        print("*" * 10, "=" * 10, "*" * 10)
        print("Loading dataloader for Segmentation task")
        
        train_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

        val_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

        test_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)
        # print number of keys in the dictionary
        print("Train Number of classes: ", len(train_class_stats.keys()))

        val_class_stats = val_dataset._get_class_stats()
        print("val class stats: ", val_class_stats)
        # print number of keys in the dictionary
        print("Val Number of classes: ", len(val_class_stats.keys()))

        
        # Use a partial function or lambda to pass the frames_per_clip argument
        collate_fn_with_args = partial(window_collate_fn, frames_per_clip=args.dataloader_params["observation_window"])

        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True, collate_fn=collate_fn_with_args)
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))


    elif (args.dataloader_params["task"] == 'cpr_quality'):
        print("*" * 10, "=" * 10, "*" * 10)
        print("Loading dataloader for CPR quality task")
        
        train_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=None, transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

        val_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=None, transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

        test_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=None, transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)
        # print number of keys in the dictionary
        print("Train Number of classes: ", len(train_class_stats.keys()))

        val_class_stats = val_dataset._get_class_stats()
        print("val class stats: ", val_class_stats)
        # print number of keys in the dictionary
        print("Val Number of classes: ", len(val_class_stats.keys()))

        

        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False) # temporary set shuffle=False
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))
        
    return train_loader, val_loader, test_loader, train_class_stats, val_class_stats





def preprocess(x, modality, backbone, device, task='classification'):
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']

    if task == 'segmentation':
        majority_label, _ = torch.mode(label, dim=1)  # [batch_size], mode returns (values, indices)
        label = majority_label


    if ('video' in modality and 'smartwatch' in modality and task == 'cpr_quality'):
        # extract resnet50 features
        frames = x['frames']
        frames = frames.to(device)

        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        # concatenate all features
        
        feature = {'frames': frames, 'smartwatch': smartwatch}
        label =  x['depth_sensor'].float()
        
        print("Special case for CPR quality task")
        return feature, None, label
    
    elif ('smartwatch' in modality and task == 'cpr_quality'):
        # smartwatch features are already extracted
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        feature = {'smartwatch': smartwatch}
        label = x['depth_sensor'].float()
        return feature, None, label
        
    elif('video' in modality):
        feature = None
        x = x['frames']
        # extract resnet50 features
        x = x.to(device)
        x = backbone.extract_resnet(x)
        feature = x
        


    elif ( 'audio' in modality and  'resnet' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet'].float()
        resnet = resnet.to(device)

        audio = x['audio']
        audio = audio.to(device)
        audio = backbone.extract_mel_spectrogram(audio, multimodal=True)
        
        feature = torch.cat((resnet, audio), dim=1).float()

    elif ( 'flow' in modality and  'rgb' in modality and  'smartwatch' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        smartwatch = x['smartwatch'].float()

        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        # concatenate all features
        feature = torch.cat((flow, rgb, smartwatch), dim=-1).float()
        
    elif ( 'flow' in modality and  'rgb' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        feature = torch.cat((flow, rgb), dim=-1).float()

    elif ('resnet' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, smartwatch), dim=-1).float()

    elif ('resnet' in modality and 'resnet_exo' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet'].float()
        resnet_exo = x['resnet_exo'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, resnet_exo, smartwatch), dim=-1).float()


    elif ('resnet' in modality and 'resnet_exo' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet'].float()
        resnet_exo = x['resnet_exo'].float()
        feature = torch.cat((resnet, resnet_exo), dim=-1).float()


    elif ('resnet' in modality):
        # resnet50 features are already extracted
        feature = x['resnet'].float()

    elif ('resnet_exo' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_exo'].float()

    elif ('rgb' in modality):
        # I3D features are already extracted
        feature = x['rgb'].float()

    elif ('flow' in modality):
        # I3D features are already extracted
        feature = x['flow'].float()

    elif ('audio' in modality):
        # Audio features are already extracted

        # Example batch of audio clips (batch, samples, channels)
        audio_clips = x['audio']  # Assume shape [batch, samples, channels]
        audio_clips = audio_clips.to(device)
        feature = backbone.extract_mel_spectrogram(audio_clips)

    elif ('smartwatch' in modality):
        # Audio features are already extracted
        smartwatch = x['smartwatch'].float()
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        feature = smartwatch
        
    feature_size = feature.shape[-1]

    if(feature is not None):
        feature = feature.to(device)
        label = label.to(device)

    return feature, feature_size, label

