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

import numpy as np
from collections import Counter


from ego_rate_estimator.cpr_rate_detection_video import init_models, ego_rate_detect,ego_rate_detect_cached
from ego_depth_estimator.supervised_depth_estimate_window_ego import ego_depth_estimator_cached
from smartwatch_rate_estimator.cpr_rate_detection_smartwatch import smartwatch_rate_detect, get_gt_cpr_rate, get_gt_cpr_depth
from smartwatch_depth_estimator.cpr_depth_detection_smartwatch import smartwatch_depth_estimator, smartwatch_depth_estimator_inference


# detect function for train fusion cpr rate estimation
def detect_depth(frames, smartwatch, gt, window_size,window_start_idx,window_end_idx, video_id, CACHE_DIR, midas_model, midas_transform, device, SCALE_FACTOR, smartwatch_model, smartwatch_optimizer, smartwatch_criterion, MODE="train"):
    """
    Slide a tumbling window and compute:
      - smartwatch_depth per window
      - ego_video_depth per window
      - gt_depth per window
    Returns three 1D numpy arrays: (sw_rates, vid_rates, gt_rates)
    """
    sw_depths, vid_depths, gt_depths = [], [], []
    total = frames.shape[0]

    print("\n\n" + "=" * 50)
    print(f"Processing video {video_id} with {total} frames, smartwatch data length {len(smartwatch)}, and gt data length {len(gt)}")
    print(f"In mode: {MODE}")

    print("-*" * 20)
    print(f"[CPR DEPTH DETECTION]:Processing window {window_start_idx}-{window_end_idx} for video {video_id}...")

    try:
            
        win_frames = frames
        win_smart = smartwatch
        win_gt = gt

        print(f"Window size: {len(win_frames)} frames, {len(win_smart)} smartwatch data points, {len(win_gt)} GT data points")
# 
        # v_rate = ego_rate_detect(win_frames, video_id)
        v_depth, v_depth_3d, v_abs  = ego_depth_estimator_cached(win_frames, video_id, window_start_idx, window_end_idx, CACHE_DIR,  midas_model, midas_transform, device, SCALE_FACTOR)

        if MODE == "train":
            s_depth = smartwatch_depth_estimator(win_smart, win_gt, smartwatch_model, smartwatch_optimizer, smartwatch_criterion)
        else:
            s_depth = smartwatch_depth_estimator_inference(win_smart, win_gt, smartwatch_model, None, None)
        
        _,_,g_depth = get_gt_cpr_depth(win_gt)

        print(f"depths for window  - Video: {v_depth}, SWDepth: {s_depth}, GT: {g_depth}")
        # skip if any rate is None

        # if v_depth is None, set to 0
        if v_depth is None:
            v_depth = 0.0
        if s_depth is None:
            s_depth = 0.0
        if g_depth is None:
            g_depth = 0.0

        # ensure Python floats
        try:
            v_depth = float(v_depth)
            v_abs = float(v_abs)
            s_depth = float(s_depth)
            g_depth = float(g_depth)
        except (TypeError, ValueError):
            print("⚠️ Invalid depth values, skipping this window.")

        sw_depths.append(s_depth)
        vid_depths.append(v_abs) # using v_abs as the ego depth
        gt_depths.append(g_depth)

        print(f"window:,sw_depths:{s_depth:.2f},ego_rate:{v_depth:.2f},gt_rate:{g_depth:.2f}")

    except Exception as e:
        print(f"Error processing window  for video {video_id}: {e}")
        # stack trace for debugging
        import traceback
        traceback.print_exc()

    print("-*" * 20)
    return np.array(sw_depths), np.array(vid_depths), np.array(gt_depths)










# detect function for train fusion cpr rate estimation
def detect_rate(frames, smartwatch, gt, window_size, window_start_idx, window_end_idx, video_id, CACHE_DIR):
    """
    Slide a tumbling window and compute:
      - smartwatch_rate per window
      - ego_video_rate per window
      - gt_rate per window
    Returns three 1D numpy arrays: (sw_rates, vid_rates, gt_rates)
    """
    sw_rates, vid_rates, gt_rates = [], [], []
    total = frames.shape[0]

    print(f"Processing video {video_id} with {total} frames, smartwatch data length {len(smartwatch)}, and gt data length {len(gt)}")


    print("-*" * 20)
    print(f"[CPR RATE DETECTION]: Processing window for video {video_id}...")

    try:
            
        win_frames = frames
        win_smart = smartwatch
        win_gt = gt

        print(f"Window size: {len(win_frames)} frames, {len(win_smart)} smartwatch data points, {len(win_gt)} GT data points")
# 
        # v_rate = ego_rate_detect(win_frames, video_id)
        v_rate = ego_rate_detect_cached(win_frames, video_id,  window_start_idx, window_end_idx, CACHE_DIR)
        s_rate = smartwatch_rate_detect(win_smart)
        g_rate = get_gt_cpr_rate(win_gt)

        print(f"Rates for window - Video: {v_rate}, Smartwatch: {s_rate}, GT: {g_rate}")
        # skip if any rate is None


        # if v_rate is None, set to 0
        if v_rate is None:
            v_rate = 0.0
        if s_rate is None:
            s_rate = 0.0
        if g_rate is None:
            g_rate = 0.0

        # ensure Python floats
        try:
            v_rate = float(v_rate)
            s_rate = float(s_rate)
            g_rate = float(g_rate)
        except (TypeError, ValueError):
            print("⚠️ Invalid rate values, skipping this window.")


        sw_rates.append(s_rate)
        vid_rates.append(v_rate)
        gt_rates.append(g_rate)

        print(f"window,sw_rate:{s_rate:.2f},ego_rate:{v_rate:.2f},gt_rate:{g_rate:.2f}")

    except Exception as e:
        print(f"Error processing window for video {video_id}: {e}")
        # stack trace for debugging
        import traceback
        traceback.print_exc()

    print("-*" * 20)
    return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)





# # detect function for test fusion cpr rate estimation
# def detect_rate(frames, smartwatch, gt, window_size, video_id):
#     """Compute per-window rates for one clip."""
#     sw_rates, vid_rates, gt_rates = [], [], []
#     total = frames.shape[0]
#     for start in range(0, total, window_size):
#         end = start + window_size
#         if end > total:
#             break
#         win_frames = frames[start:end]
#         win_sw = smartwatch[start:end]
#         win_gt = gt[start:end]

#         # estimates
#         # v = ego_rate_detect(win_frames, video_id)
#         v = ego_rate_detect_cached(win_frames, video_id, start, end, cache_dir=CACHE_DIR)
#         s = smartwatch_rate_detect(win_sw)
#         g = get_gt_cpr_rate(win_gt)

#         print(f"Window {start}-{end}: Video rate: {v}, Smartwatch rate: {s}, GT rate: {g}")

#         try:
#             v = float(v); s = float(s); g = float(g)
#         except:
#             continue
#         if np.isnan(v) or np.isnan(s) or np.isnan(g):
#             continue

#         sw_rates.append(s)
#         vid_rates.append(v)
#         gt_rates.append(g)
#     return np.array(sw_rates), np.array(vid_rates), np.array(gt_rates)
















def trim_pre_cpr(sw_rates, vid_rates, gt_rates):
    """
    Automatically finds the modal GT rate (i.e. the most frequent rate)
    and chops off all leading windows until we hit that rate.

    Returns:
      sw2, vid2, gt2, fused2, start_idx
    """
    # to guard against floating‐point artifacts, round to nearest integer
    gt_int = np.rint(gt_rates).astype(int)

    # find the most common GT value
    mode_val, _ = Counter(gt_int).most_common(1)[0]

    # find the first window where GT == mode_val
    idxs = np.where(gt_int == mode_val)[0]
    if len(idxs) == 0:
        # no “normal” windows found; return everything
        start_idx = 0
    else:
        start_idx = idxs[0]

    # slice all four series from that point onward
    return (
        np.asarray(sw_rates)[start_idx:],
        np.asarray(vid_rates)[start_idx:],
        np.asarray(gt_rates)[start_idx:],
        int(start_idx)
    )

def preprocess(x, modality, backbone, device, task='classification'):
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']

    if task == 'segmentation':
        majority_label, _ = torch.mode(label, dim=1)  # [batch_size], mode returns (values, indices)
        label = majority_label


    if ('video' in modality and 'smartwatch' in modality and task == 'cpr_quality'):
        # extract resnet50 features
        frames = x['video']
        frames = frames.to(device)

        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        # concatenate all features
        
        feature = {'video': frames, 'smartwatch': smartwatch}
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
        label = x['depth_sensor'].float()  # Assuming depth sensor data is the label
        
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
        
        train_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

        val_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

        test_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path='',
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

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


