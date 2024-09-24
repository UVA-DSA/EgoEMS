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
    
    cmd_args = parser.parse_args()
    
    print(f"Job ID: {cmd_args.job_id}")

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

    # Access the parsed arguments
    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)  # adjust parameters as needed

    # Load the best model
    model.load_state_dict(torch.load(f'./checkpoints/{args.learning_params["best_chkpoint"]}'))


    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader = eee_get_dataloaders(args)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f'./results/{cmd_args.job_id}'
 
    # create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


# # Test the model
    eee_test_model(model, train_loader, criterion, device, wandb_logger, 0, results_dir)
