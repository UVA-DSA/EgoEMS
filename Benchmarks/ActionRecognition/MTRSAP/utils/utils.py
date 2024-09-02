#import from models folder transtcn
from models.transtcn import TransformerModel
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(args)
    model.to(device)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_params["lr"], weight_decay=args.learning_params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
        
    return model, optimizer, criterion, device


# add wandb logging
def train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0
    for i, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(videos)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logger.log({"train_loss": loss.item()})
        print(f"Batch: {i}, Loss: {loss.item()}")
    return total_loss / len(train_loader)


# validate the model 
def validate(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (videos, labels) in enumerate(val_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            output = model(videos)
            loss = criterion(output, labels)
            total_loss += loss.item()
            logger.log({"val_loss": loss.item()})
    return total_loss / len(val_loader)


# test the model
def test_model(model, test_loader, criterion, device, logger, epoch, results_dir):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    

    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            output = model(videos)
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


# return train,val,test dataloaders
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


