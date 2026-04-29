#import from models folder transtcn
from models.transtcn import TransformerModel, MultimodalFusion
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from EgoEMS.EgoEMS import  WindowEgoEMSDataset, EgoEMSDataset, collate_fn, transform, window_collate_fn
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

fusion = MultimodalFusion()

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=None, num_classes=None, class_counts=None, weights=None):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

        if weights is not None:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        elif class_counts is not None:
            class_count_tensor = torch.as_tensor(class_counts, dtype=torch.float32)
            positive_mask = class_count_tensor > 0
            weight_tensor = torch.zeros_like(class_count_tensor)
            if positive_mask.any():
                positive_counts = class_count_tensor[positive_mask]
                positive_weights = (1 - beta) / (1 - beta ** positive_counts)
                positive_weights = positive_weights / positive_weights.mean()
                weight_tensor[positive_mask] = positive_weights
        else:
            raise ValueError("Either class_counts or weights must be provided.")

        self.register_buffer("weights", weight_tensor)

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, labels, weight=self.weights.to(logits.device))
        return loss


def _aligned_class_counts(class_names, class_stats):
    class_stats = dict(class_stats) if class_stats is not None else {}
    return torch.tensor([float(class_stats.get(class_name, 0)) for class_name in class_names], dtype=torch.float32)


def _build_loss_weights(class_counts, beta=0.99, weight_power=0.5, max_loss_weight=None):
    class_counts = torch.as_tensor(class_counts, dtype=torch.float32)
    weights = torch.zeros_like(class_counts)
    positive_mask = class_counts > 0

    if positive_mask.any():
        positive_counts = class_counts[positive_mask]
        positive_weights = (1 - beta) / (1 - beta ** positive_counts)
        if weight_power != 1.0:
            positive_weights = positive_weights.pow(weight_power)
        positive_weights = positive_weights / positive_weights.mean()
        if max_loss_weight is not None:
            positive_weights = torch.clamp(positive_weights, max=max_loss_weight)
        weights[positive_mask] = positive_weights

    return weights


def _build_train_sampler(dataset, args):
    imbalance_params = getattr(args, "imbalance_params", {})
    if not imbalance_params.get("use_weighted_sampler", False):
        return None

    sample_label_ids = []
    if hasattr(dataset, "get_sample_label_ids"):
        sample_label_ids = dataset.get_sample_label_ids()

    if len(sample_label_ids) == 0:
        return None

    label_tensor = torch.tensor(sample_label_ids, dtype=torch.long)
    num_classes = len(args.dataloader_params["keysteps"])
    class_counts = torch.bincount(label_tensor, minlength=num_classes).float()

    sampler_power = imbalance_params.get("sampler_power", 1.0)
    per_sample_counts = class_counts[label_tensor]
    sample_weights = torch.pow(per_sample_counts, -sampler_power).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_label_ids),
        replacement=imbalance_params.get("sampler_replacement", True),
    )

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(args)
    model.to(device)

    class_names = list(args.dataloader_params["keysteps"].keys())
    num_classes = len(class_names)
    train_class_counts = _aligned_class_counts(class_names, args.dataloader_params["train_class_stats"])
    val_class_counts = _aligned_class_counts(class_names, args.dataloader_params["val_class_stats"])
    imbalance_params = getattr(args, "imbalance_params", {})

    print("Training class counts: ", args.dataloader_params["train_class_stats"])
    print("Validation class counts: ", val_class_counts)
    print("Aligned class counts: ", train_class_counts, len(train_class_counts))
           
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_params["lr"], weight_decay=args.learning_params["weight_decay"])

    if imbalance_params.get("use_class_balanced_loss", True):
        loss_weights = _build_loss_weights(
            train_class_counts,
            beta=imbalance_params.get("loss_beta", 0.99),
            weight_power=imbalance_params.get("loss_weight_power", 0.5),
            max_loss_weight=imbalance_params.get("max_loss_weight", 5.0),
        )
        print("Loss weights: ", loss_weights)
        criterion = ClassBalancedLoss(num_classes=num_classes, weights=loss_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    return model, optimizer, criterion, device


def preprocess(x, modality, backbone, device, task='classification'):
    global fusion
    # print("-*" * 10, "Preprocessing", "*" * 10, "=" * 10)
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']
    # print(f"\nSubject ID: {x['subject_id']}, Trial ID: {x['trial_id']}, Start Frame: {x['start_frame']}, End Frame: {x['end_frame']}, Start Time: {x['start_t']}, End Time: {x['end_t']}")

    if task == 'segmentation':
        majority_label, _ = torch.mode(label, dim=1)  # [batch_size], mode returns (values, indices)
        label = majority_label


    if('video' in modality):
        feature = None
        x = x['frames']
        # extract resnet50 features
        x = x.to(device)
        x = backbone.extract_resnet(x)
        feature = x

    elif ( 'audio' in modality and  'resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        smartwatch = x['smartwatch'].float()
        smartwatch = smartwatch.to(device)
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        # print("Audio feature shape: ", audio_feature.shape)
        # print("Smartwatch feature shape: ", smartwatch.shape)
        # print("Resnet feature shape: ", resnet.shape)

        fusion = fusion.to(device)
        fused = fusion(audio_feature, resnet, smartwatch)  # [B, T_common, D_total]


        feature = fused.float()

        # feature = torch.cat((resnet, audio_feature, smartwatch), dim=-1).float()

    elif ( 'audio' in modality and  'resnet_ego' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        # audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        audio_feature = backbone.extract_mel_spectrogram(audio, multimodal=True) # for mel spectrogram features

        feature = torch.cat((resnet, audio_feature), dim=1).float()

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

    elif ('resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, smartwatch), dim=-1).float()

    elif ('resnet_ego' in modality and 'resnet_exo' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, resnet_exo, smartwatch), dim=-1).float()


    elif ('resnet_ego' in modality and 'resnet_exo' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        feature = torch.cat((resnet, resnet_exo), dim=-1).float()


    elif ('resnet_ego' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_ego'].float()

    elif ('resnet_exo' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_exo'].float()

    elif ('clip_ego' in modality):
        # resnet50 features are already extracted
        feature = x['clip_ego'].float()
        # print("Clip ego feature shape: ", feature.shape)
    elif ('clip_exo' in modality):
        # resnet50 features are already extracted
        feature = x['clip_exo'].float()
        # print("Clip exo feature shape: ", feature.shape)

    elif ('clip_ego' in modality and 'clip_exo' in modality):
        # resnet50 features are already extracted
        feature = torch.cat((x['clip_ego'].float(), x['clip_exo'].float()), dim=-1)
        # print("Clip ego and exo feature shape: ", feature.shape)

    elif ('rgb' in modality):
        # I3D features are already extracted
        feature = x['rgb'].float()

    elif ('flow' in modality):
        # I3D features are already extracted
        feature = x['flow'].float()

    # elif ('audio' in modality):
    #     # Audio features are already extracted

    #     # Example batch of audio clips (batch, samples, channels)
    #     audio_clips = x['audio']  # Assume shape [batch, samples, channels]
    #     audio_clips = audio_clips.to(device)
    #     feature = backbone.extract_mel_spectrogram(audio_clips)

    elif ('smartwatch' in modality):
        # Audio features are already extracted
        smartwatch = x['smartwatch'].float()
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        feature = smartwatch

    elif ('audio' in modality): # uncomment this if you want to use wav2vec features
        audio_clips = x['audio']  # Assume shape [batch, samples, channels]
        audio_clips = audio_clips.to(device)
        # feature = backbone.extract_wav2vec_features(audio_clips)
        feature = backbone.extract_mel_spectrogram(audio_clips)

        # print("Wav2Vec feature shape: ", feature.shape)

    feature_size = feature.shape[-1]
    # print("Feature shape: ", feature.shape, "\n")

    if(feature is not None):
        feature = feature.to(device)
        label = label.to(device)

    return feature, feature_size, label


# add wandb logging
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    logger,
    modality,
    task='classification',
    epoch=None,
    log_batch_metrics=False,
    detailed_stdout=False,
):
    model.train()
    total_loss = 0
    valid_batches = 0
    for i, batch in enumerate(train_loader):

        try:
            input,feature_size, label = preprocess(batch, modality, model, device, task=task)

            # get more info about input
            keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
            keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
            start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
            end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
            start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
            end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
            subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
            trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
            window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
            window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

            if detailed_stdout:
                print("=" * 10, "-" * 10, "=" * 10)
                if task == 'segmentation':
                    print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                    print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")
                else:
                    print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                    print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")


                    # ←—— ADDED CHECK ———→
            # if the time-dimension is zero, skip this batch
            # (inputs.shape == [B, T, F] or [B, C, T] depending on your preprocess)
            if input.size(1) == 0: 
                print(f"Skipping batch {i}: feature sequence : {input.shape}")
                continue

            if torch.isnan(input).any():
                print(f"⚠️ Skipping batch {i} because NaN")
                continue

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            if detailed_stdout and i % 1 == 0:
                print("\n")
                print("*" * 10, "=" * 10, "*" * 10)
                print(f"Pred: {torch.argmax(output, dim=1)} GT: {label}")
                if log_batch_metrics and epoch is not None:
                    logger.log({
                        "epoch": epoch,
                        "train_step": epoch * len(train_loader) + i,
                        "train/batch_loss": float(loss.item()),
                    })
                print(f"Batch: {i}, Loss: {loss.item()}")
                print("*" * 10, "=" * 10, "*" * 10)
                print("\n")
            elif log_batch_metrics and epoch is not None:
                logger.log({
                    "epoch": epoch,
                    "train_step": epoch * len(train_loader) + i,
                    "train/batch_loss": float(loss.item()),
                })
            # break
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            # print(f"Batch data: {batch}")
            continue

    if valid_batches == 0:
        raise RuntimeError("No valid training batches were processed.")

    return total_loss / valid_batches


# validate the model 
def validate(model, val_loader, criterion, device, logger, modality, task='classification', epoch=None, log_batch_metrics=False, detailed_stdout=False):
    model.eval()
    total_loss = 0
    valid_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                input,feature_size, label = preprocess(batch, modality, model, device, task=task)

                # check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue

                output = model(input)
                loss = criterion(output, label)
                total_loss += loss.item()
                valid_batches += 1
                if log_batch_metrics and epoch is not None and i % 100 == 0:
                    logger.log({
                        "epoch": epoch,
                        "val_step": epoch * len(val_loader) + i,
                        "val/batch_loss": float(loss.item()),
                    })
            # break
            
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                if detailed_stdout:
                    print(f"Batch data: {batch}")
                continue

    if valid_batches == 0:
        raise RuntimeError("No valid validation batches were processed.")

    return total_loss / valid_batches


# test the model
def test_model(model, test_loader, criterion, device, logger, epoch, results_dir, modality, task='classification', detailed_stdout=False):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                input,feature_size, label = preprocess(batch, modality, model, device, task=task)
                if detailed_stdout:
                    print("=" * 10, "-" * 10, "=" * 10)
                    print(f"[TEST] Batch: {i}")

                # check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                # get more info about input
                keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
                keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
                start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
                end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
                start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
                end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
                subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
                trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
                window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
                window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

                if detailed_stdout:
                    if task == 'segmentation':
                        print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                        print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")
                    else:
                        print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                        print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")  
                
                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue
                
                output = model(input)
                pred = torch.argmax(output, dim=1)
                if detailed_stdout:
                    print(f"Model Pred: {pred.item()}")

                gt.append(label.item())
                preds.append(pred.item())

                preds_detail.append({
                    "keystep_label": keystep_label,
                    "keystep_id": keystep_id.tolist(),
                    "start_frame": start_frame.tolist(),
                    "end_frame": end_frame.tolist(),
                    "start_t": start_t.tolist(),
                    "end_t": end_t.tolist(),
                    "window_start_frame": window_start_frame.item(),
                    "window_end_frame": window_end_frame.item(),
                    "subject_id": subject_id[0],
                    "trial_id": trial_id[0],
                    "pred_keystep_id": pred.item(),
                    "all_preds": output.tolist()
                })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                if detailed_stdout:
                    print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    if len(gt) == 0:
        raise RuntimeError("No valid test batches were processed.")

    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro', zero_division=0)
    recall = recall_score(gt, preds, average='macro', zero_division=0)
    f1 = f1_score(gt, preds, average='macro', zero_division=0)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "epoch": int(epoch)
    }
    # Log metrics to wandb
    logger.log({
        "epoch": int(epoch),
        "test/accuracy": float(results["accuracy"]),
        "test/precision": float(results["precision"]),
        "test/recall": float(results["recall"]),
        "test/f1": float(results["f1"]),
    })
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])

    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["keystep_label", "keystep_id", "start_frame", "end_frame", "start_t", "end_t","window_start_frame","window_end_frame", "subject_id", "trial_id", "pred_keystep_id","all_preds"])
        for pred in preds_detail:
            writer.writerow([pred["keystep_label"], pred["keystep_id"], pred["start_frame"], pred["end_frame"], pred["start_t"], pred["end_t"], pred["window_start_frame"],pred["window_end_frame"], pred["subject_id"], pred["trial_id"], pred["pred_keystep_id"], pred["all_preds"]])
    return results



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



''' ***** EGOEXOEMS DATASET ***** '''


# add wandb logging
def eee_train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        i3d_rgb_features = batch['rgb']
        i3d_flow_features = batch['flow']

        # move to device
        i3d_rgb_features = i3d_rgb_features.to(device)
        i3d_flow_features = i3d_flow_features.to(device)

        # get labels
        labels = batch['keystep_id']
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(i3d_rgb_features)


        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 1 == 0:
            print("\n ***** ")
            print(batch['frames'].shape, batch['audio'].shape, batch['flow'].shape, batch['rgb'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'],batch['start_t'], batch['end_t'],  batch['subject_id'], batch['trial_id'])
            print(f"Pred: {torch.argmax(output, dim=1)} GT: {labels}")
            logger.log({"train_loss": loss.item()})
            print(f"Batch: {i}, Loss: {loss.item()}")
            print(" ***** \n")

    return total_loss / len(train_loader)


# validate the model 
def eee_validate(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            i3d_rgb_features = batch['rgb']
            i3d_flow_features = batch['flow']

            # move to device
            i3d_rgb_features = i3d_rgb_features.to(device)
            i3d_flow_features = i3d_flow_features.to(device)

            # get labels
            labels = batch['keystep_id']
            labels = labels.to(device)

            output = model(i3d_rgb_features)

            loss = criterion(output, labels)
            total_loss += loss.item()
            if i % 1 == 0:
                logger.log({"val_loss": loss.item()})

    return total_loss / len(val_loader)


# test the model
def eee_test_model(model, test_loader, criterion, device, logger, epoch, results_dir):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            i3d_rgb_features = batch['rgb']
            i3d_flow_features = batch['flow']

            # move to device
            i3d_rgb_features = i3d_rgb_features.to(device)
            i3d_flow_features = i3d_flow_features.to(device)

            # get labels
            labels = batch['keystep_id']
            labels = labels.to(device)

            output = model(i3d_rgb_features)
            pred = torch.argmax(output, dim=1)
            gt.append(labels.item())
            preds.append(pred.item())
    
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    # Log metrics to wandb
    logger.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "epoch": epoch
    })
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not os.path.isfile(metrics_path):
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
        writer.writerow([epoch, accuracy, precision, recall, f1])
    
    return accuracy




# return train,val,test dataloaders using the EgoEMSDataset class
def eee_get_dataloaders(args):
    train_sampler = None
    
    if(args.dataloader_params["task"] == 'classification'):
        print("*" * 10, "=" * 10, "*" * 10)
        print("Loading dataloader for Classification task")

        train_dataset = EgoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])

        val_dataset = EgoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])

        test_dataset = EgoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])


        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)
        # print number of keys in the dictionary
        print("Train Number of classes: ", len(train_class_stats.keys()))

        val_class_stats = val_dataset._get_class_stats()
        print("val class stats: ", val_class_stats)
        # print number of keys in the dictionary
        print("Val Number of classes: ", len(val_class_stats.keys()))

        train_sampler = _build_train_sampler(train_dataset, args)

        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.dataloader_params["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))
    
    elif (args.dataloader_params["task"] == 'segmentation'):
        print("*" * 10, "=" * 10, "*" * 10)
        print("Loading dataloader for Segmentation task")
        
        train_dataset = WindowEgoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])

        val_dataset = WindowEgoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])

        test_dataset = WindowEgoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                        data_base_path=args.dataloader_params["data_base_path"],
                                        fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"],
                                        classes=args.dataloader_params["classes"])

        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)
        # print number of keys in the dictionary
        print("Train Number of classes: ", len(train_class_stats.keys()))

        val_class_stats = val_dataset._get_class_stats()
        print("val class stats: ", val_class_stats)
        # print number of keys in the dictionary
        print("Val Number of classes: ", len(val_class_stats.keys()))

        test_class_stats = test_dataset._get_class_stats()
        print("test class stats: ", test_class_stats)
        # print number of keys in the dictionary
        print("Test Number of classes: ", len(test_class_stats.keys()))

        train_sampler = _build_train_sampler(train_dataset, args)
        
        # Use a partial function or lambda to pass the frames_per_clip argument
        collate_fn_with_args = partial(window_collate_fn, frames_per_clip=args.dataloader_params["observation_window"])

        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.dataloader_params["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn_with_args,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)

        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(test_dataset))

    if train_sampler is not None:
        print("Using weighted sampler for training")

    return train_loader, val_loader, test_loader, train_class_stats, val_class_stats
