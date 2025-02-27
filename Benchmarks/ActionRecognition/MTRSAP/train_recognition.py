from utils.utils import *
from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models
from datautils.ems import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime

import argparse
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


torch.manual_seed(0)

if __name__ == "__main__":

    # get cmd line args
    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--job_id', type=str, help='SLURM job ID')
    
    cmd_args = parser.parse_args()
    
    print(f"Job ID: {cmd_args.job_id}")

    args = DefaultArgsNamespace()

    wandb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="EgoExoEMS",
        group="Keystep Recognition",
        # mode="disabled",
        name="Rebuttal - ego,exo,imu",
        notes="",
        config={
        "args": args,
        }
    )

    keysteps = args.dataloader_params['keysteps']
    out_classes = len(keysteps)

    modality = args.dataloader_params['modality']
    print("Modality: ", modality)
    print("Num of classes: ", out_classes)

    window = args.dataloader_params['observation_window']
    print("Window: ", window)

    task = args.dataloader_params['task']
    print("Task: ", task)
    

    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)
    args.dataloader_params['train_class_stats'] = train_class_stats
    args.dataloader_params['val_class_stats'] = val_class_stats
    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    model = model.to(device)

    # Find feature dimension
    feature,feature_size,label = preprocess(next(iter(train_loader)), args.dataloader_params['modality'], model, device)
    print("Feature size: ", feature_size)

    print("Reinitializing model with feature size")

    args.transformer_params['input_dim'] = feature_size
    args.transformer_params['output_dim'] = out_classes

    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    model = model.to(device)
    scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)  # adjust parameters as needed

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f'./results/{cmd_args.job_id}'
    chkpoint_dir = f'./checkpoints/{cmd_args.job_id}'

    # create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create checkpoint directory if not exists
    if not os.path.exists(chkpoint_dir):
        os.makedirs(chkpoint_dir)
    
    min_val_loss = float('inf')

    # # Train the model
    for epoch in range(args.learning_params["epochs"]):
        print("*"*10, "="*10, "*"*10)
        print(f"Epoch: {epoch}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, wandb_logger, modality=modality, task=task)
        wandb_logger.log({"avg_train_loss": train_loss, "epoch": epoch})
        val_loss = validate(model, val_loader, criterion, device, wandb_logger, modality=modality, task=task)
        wandb_logger.log({"avg_val_loss": val_loss, "epoch": epoch})

        # save checkpoints if validation loss is minimum 
        # if val_loss < min_val_loss:
        #     min_val_loss = val_loss
        torch.save(model.state_dict(), f'{chkpoint_dir}/val_best_model.pt')


        scheduler.step()
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        results = test_model(model, test_loader, criterion, device, wandb_logger, epoch, results_dir, modality=modality, task=task)
        print(f"Results: {results}")
        
        print("*"*10, "="*10, "*"*10)
        
