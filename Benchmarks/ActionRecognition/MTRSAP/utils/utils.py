#import from models folder transtcn
from models.transtcn import TransformerModel
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from EgoExoEMS.EgoExoEMS import EgoExoEMSDataset, collate_fn, transform

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(args)
    model.to(device)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_params["lr"], weight_decay=args.learning_params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
        
    return model, optimizer, criterion, device


def preprocess(x, modality, backbone, device):
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']
    if('video' in modality):
        feature = None
        x = x['frames']
        # extract resnet50 features
        x = x.to(device)
        x = backbone(x)
        feature = x
    



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

    elif ('resnet' in modality):
        # resnet50 features are already extracted
        feature = x['resnet'].float()

    elif ('rgb' in modality):
        # I3D features are already extracted
        feature = x['rgb'].float()

    elif ('flow' in modality):
        # I3D features are already extracted
        feature = x['flow'].float()

    elif ('audio' in modality):
        # Audio features are already extracted
        feature = x['audio'].float()

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


# add wandb logging
def train_one_epoch(model, train_loader, criterion, optimizer, device, logger, modality):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input,feature_size, label = preprocess(batch, modality, model.extract_resnet, device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print("\n")
            print("*" * 10, "=" * 10, "*" * 10)
            print(f"Pred: {torch.argmax(output, dim=1)} GT: {label}")
            logger.log({"train_loss": loss.item()})
            print(f"Batch: {i}, Loss: {loss.item()}")
            print("*" * 10, "=" * 10, "*" * 10)
            print("\n")
        # break
        
    return total_loss / len(train_loader)


# validate the model 
def validate(model, val_loader, criterion, device, logger, modality):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input,feature_size, label = preprocess(batch, modality, model.extract_resnet, device)
            output = model(input)
            loss = criterion(output, label)
            total_loss += loss.item()
            if i % 100 == 0:
                logger.log({"val_loss": loss.item()})
            # break
            
            
    return total_loss / len(val_loader)


# test the model
def test_model(model, test_loader, criterion, device, logger, epoch, results_dir, modality):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input,feature_size, label = preprocess(batch, modality, model.extract_resnet, device)

            # get more info about input
            keystep_label = batch['keystep_label']
            keystep_id = batch['keystep_id']
            start_frame = batch['start_frame']
            end_frame = batch['end_frame']
            start_t = batch['start_t']
            end_t = batch['end_t']
            subject_id = batch['subject_id']
            trial_id = batch['trial_id']


            output = model(input)
            pred = torch.argmax(output, dim=1)

            gt.append(label.item())
            preds.append(pred.item())

            preds_detail.append({
                "keystep_label": keystep_label,
                "keystep_id": keystep_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_t": start_t,
                "end_t": end_t,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "pred_keystep_id": pred.item()
            })


            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not os.path.isfile(metrics_path):
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
        writer.writerow([epoch, accuracy, precision, recall, f1])
    

    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not os.path.isfile(preds_path):
            writer.writerow(["keystep_label", "keystep_id", "start_frame", "end_frame", "start_t", "end_t", "subject_id", "trial_id", "pred_keystep_id"])
        for pred in preds_detail:
            writer.writerow([pred["keystep_label"], pred["keystep_id"], pred["start_frame"], pred["end_frame"], pred["start_t"], pred["end_t"], pred["subject_id"], pred["trial_id"], pred["pred_keystep_id"]])


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




# return train,val,test dataloaders using the EgoExoEMSDataset class
def eee_get_dataloaders(args):

    train_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
                                    data_base_path='',
                                    fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

    val_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
                                    data_base_path='',
                                    fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])

    test_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
                                    data_base_path='',
                                    fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"])



    # Create DataLoaders for training and validation subsets
    train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

    print("train dataset size: ", len(train_dataset))
    print("val dataset size: ", len(val_dataset))
    print("test dataset size: ", len(test_dataset))

    return train_loader, val_loader, test_loader


