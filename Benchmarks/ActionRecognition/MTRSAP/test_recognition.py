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
import json
import os

import argparse


torch.manual_seed(0)


def _set_active_classes(args, classes, all_keysteps):
    active_classes = [class_name for class_name in classes if class_name in all_keysteps]
    missing_config_classes = [class_name for class_name in classes if class_name not in all_keysteps]
    if missing_config_classes:
        print(f"[Warning] Config classes missing from keysteps and ignored: {missing_config_classes}")
    if not active_classes:
        raise ValueError("No active classes remain after applying the configured class list.")

    args.dataloader_params["classes"] = active_classes
    args.dataloader_params["keysteps"] = {
        class_name: all_keysteps[class_name] for class_name in active_classes
    }
    return args.dataloader_params["keysteps"]


def _load_checkpoint_classes(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    mapping_paths = [
        os.path.join(checkpoint_dir, filename)
        for filename in os.listdir(checkpoint_dir)
        if filename.endswith("_class_mapping.json")
    ]
    if not mapping_paths:
        return None

    mapping_paths.sort(key=os.path.getmtime, reverse=True)
    with open(mapping_paths[0], "r") as f:
        mapping = json.load(f)

    new_id_to_class = mapping.get("new_id_to_class", {})
    if not new_id_to_class:
        return None

    classes = [
        class_name for _, class_name in sorted(
            ((int(class_id), class_name) for class_id, class_name in new_id_to_class.items()),
            key=lambda item: item[0],
        )
    ]
    print(f"Loaded checkpoint class mapping from: {mapping_paths[0]}")
    return classes


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_wandb_mode(args):
    wandb_params = getattr(args, "wandb_params", {})
    if not wandb_params.get("enabled", True):
        return "disabled"

    mode = wandb_params.get("mode", "auto")
    if mode == "auto":
        return "online" if os.environ.get("WANDB_API_KEY") else "offline"
    return mode


def _build_wandb_config(args, cmd_args):
    return {
        "job_id": cmd_args.job_id,
        "cli_args": _to_serializable(vars(cmd_args)),
        "learning_params": _to_serializable(args.learning_params),
        "training_control_params": _to_serializable(args.training_control_params),
        "logging_params": _to_serializable(args.logging_params),
        "wandb_params": _to_serializable(args.wandb_params),
        "imbalance_params": _to_serializable(args.imbalance_params),
        "dataloader_params": _to_serializable(args.dataloader_params),
        "transformer_params": _to_serializable(args.transformer_params),
        "tcn_model_params": _to_serializable(args.tcn_model_params),
    }


def _init_wandb_logger(args, cmd_args, default_name):
    wandb_params = getattr(args, "wandb_params", {})
    wandb_mode = _resolve_wandb_mode(args)
    run_name = wandb_params.get("name") or default_name

    wandb_logger = wandb.init(
        project=wandb_params.get("project", "EgoExoEMS"),
        group=wandb_params.get("group", "Keystep Recognition"),
        mode=wandb_mode,
        name=run_name,
        notes=wandb_params.get("notes", ""),
        tags=wandb_params.get("tags", None),
        config=_build_wandb_config(args, cmd_args),
    )

    wandb_logger.define_metric("epoch")
    for metric_name in [
        "test/accuracy",
        "test/precision",
        "test/recall",
        "test/f1",
    ]:
        wandb_logger.define_metric(metric_name, step_metric="epoch")
    wandb_logger.summary["wandb_mode"] = wandb_mode
    return wandb_logger, wandb_mode

if __name__ == "__main__":

    # get cmd line args
    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--job_id', type=str, help='SLURM job ID')
    parser.add_argument('--modality', type=str, default=None, help='Override Modality to use for training')
    
    cmd_args = parser.parse_args()
    
    print(f"Job ID: {cmd_args.job_id}")
    print(f"Modality: {cmd_args.modality}")

    args = DefaultArgsNamespace()

    default_run_name = f"MTRSAP-test-{cmd_args.job_id or 'local'}"
    wandb_logger, wandb_mode = _init_wandb_logger(args, cmd_args, default_run_name)
    print(f"W&B mode: {wandb_mode}")

    keysteps = args.dataloader_params['keysteps']
    classes = args.dataloader_params['classes']

        # Filter keysteps using the exact classes order so label remap and class-weights stay aligned.
    keysteps = {k: keysteps[k] for k in classes if k in keysteps}
    args.dataloader_params['keysteps'] = keysteps
    
    out_classes = len(keysteps)
    out_classes = len(keysteps)

    modality = args.dataloader_params['modality']

    if cmd_args.modality is not None:
        modality = cmd_args.modality
        modality = modality.split(",") if ',' in modality else [modality]
        args.dataloader_params['modality'] = modality
        print(f"Overriding modality to: {modality}")

    print("Modality: ", modality)
    print("Num of classes: ", out_classes)

    task = args.dataloader_params['task']
    detailed_stdout = getattr(args, "logging_params", {}).get("detailed_stdout", False)
    print("Task: ", task)

    # Access the parsed arguments
    model, optimizer, criterion, device = init_model(args)# verbose_mode = args.verbose
    scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)  # adjust parameters as needed


    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)
    args.dataloader_params['train_class_stats'] = train_class_stats
    args.dataloader_params['val_class_stats'] = val_class_stats

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
    results = test_model(model, test_loader, criterion, device, wandb_logger, 0, results_dir, modality=modality, task=task, detailed_stdout=detailed_stdout)
    print(f"Results: {results}")
    wandb_logger.summary["test_results"] = results
    wandb_logger.finish()
