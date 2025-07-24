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


torch.manual_seed(0)

if __name__ == "__main__":

    # get cmd line args
    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--job_id', type=str, help='SLURM job ID')
    parser.add_argument('--modality', type=str, default=None, help='Override Modality to use for training')
    
    cmd_args = parser.parse_args()
    
    print(f"Job ID: {cmd_args.job_id}")
    print(f"Modality: {cmd_args.modality}")

    args = DefaultArgsNamespace()

    wandb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="EgoExoEMS",
        group="Keystep Recognition",
        mode="disabled",
        name="Testing on EgoExoEMS with I3D RGB Features - ICRA Model",
        notes="initial attempt ICRA model with I3D RGB features",
        config={
        "args": args,
        }
    )

    keysteps = args.dataloader_params['keysteps']
    out_classes = len(keysteps)

    modality = args.dataloader_params['modality']

    if cmd_args.modality is not None:
        modality = cmd_args.modality
        args.dataloader_params['modality'] = modality
        print(f"Overriding modality to: {modality}")

    print("Modality: ", modality)
    print("Num of classes: ", out_classes)

    task = args.dataloader_params['task']
    print("Task: ", task)

    # Access the parsed arguments
    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)  # adjust parameters as needed


    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)

    # Find feature dimension
    feature,feature_size,label = preprocess(next(iter(train_loader)), args.dataloader_params['modality'], model, device, task=task)
    print("Feature size: ", feature_size)

    print("Reinitializing model with feature size")

    args.transformer_params['input_dim'] = feature_size
    args.transformer_params['output_dim'] = out_classes

    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    model = model.to(device)

    # Load the best model
    model.load_state_dict(torch.load(args.learning_params["best_chkpoint"]), strict=False)


    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_id = args.learning_params["best_chkpoint"].split('/')[-2]


    results_dir = f'./results/model_id_{model_id}_on_{current_time}'
 
    # create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


# # Test the model
    results = test_model(model, test_loader, criterion, device, wandb_logger, 0, results_dir, modality=modality, task=task)
    print(f"Results: {results}")