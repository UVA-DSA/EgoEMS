#import from models folder transtcn
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from EgoExoEMS.EgoExoEMS import  WindowEgoExoEMSDataset, EgoExoEMSDataset, collate_fn, transform, window_collate_fn
from functools import partial
import torch.nn.functional as F
import numpy as np

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



# return train,val,test dataloaders using the VideoDataset class
def get_dataloaders(args):
    train_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=True)
    test_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=False)

    split_indices_path = f'{args.dataloader_params["base_path"]}/val_test_split_indices_fold_0{args.dataloader_params["fold"]}.npz'

    if os.path.exists(split_indices_path):
        # Load pre-existing indices
        split_data = np.load(split_indices_path)
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
    else:
        # Create new split and save the indices
        total_size = len(test_dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        val_size = int(0.5 * total_size)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        # Save the indices for later use
        np.savez(split_indices_path, val_indices=val_indices, test_indices=test_indices)
    
        # Subset datasets based on indices
    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)


    # Create DataLoaders for training and validation subsets
    train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

    print("train dataset size: ", len(train_dataset))
    print("val dataset size: ", len(val_dataset))
    print("test dataset size: ", len(test_dataset))

    return train_loader, val_loader, test_loader






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
        train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))
        
    return train_loader, val_loader, test_loader, train_class_stats, val_class_stats


