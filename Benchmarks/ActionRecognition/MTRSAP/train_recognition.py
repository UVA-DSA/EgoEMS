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
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


torch.manual_seed(0)


def warn_if_split_has_limited_coverage(split_name, class_stats, expected_classes):
    covered_classes = [class_name for class_name in expected_classes if class_stats.get(class_name, 0) > 0]
    missing_classes = [class_name for class_name in expected_classes if class_stats.get(class_name, 0) == 0]

    print(f"{split_name} split covers {len(covered_classes)}/{len(expected_classes)} target classes")
    if missing_classes:
        print(f"[Warning] {split_name} split is missing target classes: {missing_classes}")


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


def _filter_to_train_supported_classes(args, train_class_stats, all_keysteps):
    if not args.dataloader_params.get("filter_to_train_classes", False):
        return []

    min_count = args.dataloader_params.get("min_train_samples_per_class", 1)
    active_classes = list(args.dataloader_params["classes"])
    supported_classes = [
        class_name for class_name in active_classes
        if train_class_stats.get(class_name, 0) >= min_count
    ]
    dropped_classes = [
        class_name for class_name in active_classes
        if train_class_stats.get(class_name, 0) < min_count
    ]

    if not dropped_classes:
        return []
    if not supported_classes:
        raise ValueError(
            f"No classes have at least {min_count} training samples; cannot train a classifier."
        )

    print(
        f"[Info] Filtering out {len(dropped_classes)} classes with fewer than "
        f"{min_count} training samples: {dropped_classes}"
    )
    _set_active_classes(args, supported_classes, all_keysteps)
    return dropped_classes


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
        "train/avg_loss",
        "val/loss",
        "val/best_loss",
        "train/learning_rate",
        "test/accuracy",
        "test/precision",
        "test/recall",
        "test/f1",
    ]:
        wandb_logger.define_metric(metric_name, step_metric="epoch")

    wandb_logger.define_metric("train_step")
    wandb_logger.define_metric("val_step")
    wandb_logger.define_metric("train/batch_loss", step_metric="train_step")
    wandb_logger.define_metric("val/batch_loss", step_metric="val_step")
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

    wandb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="EgoExoEMS",
        group="Keystep Recognition",
        mode="online",
        name="Rebuttal - ego,exo,imu",
        notes="",
        config={
        "args": args,
        }
    )

    all_keysteps = dict(args.dataloader_params['keysteps'])
    classes = list(args.dataloader_params['classes'])
    keysteps = _set_active_classes(args, classes, all_keysteps)
    out_classes = len(keysteps)

    modality = args.dataloader_params['modality']

    if cmd_args.modality is not None:
        modality = cmd_args.modality
        modality = modality.split(",") if ',' in modality else [modality]
        args.dataloader_params['modality'] = modality
        print(f"Overriding modality to: {modality}")

    print("Modality: ", modality)
    print("Num of classes: ", out_classes)


    window = args.dataloader_params['observation_window']
    print("Window: ", window)

    task = args.dataloader_params['task']
    print("Task: ", task)
    

    dropped_train_classes = []

    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)
    args.dataloader_params['train_class_stats'] = train_class_stats
    args.dataloader_params['val_class_stats'] = val_class_stats

    dropped_train_classes = _filter_to_train_supported_classes(args, train_class_stats, all_keysteps)
    if dropped_train_classes:
        print("[Info] Rebuilding dataloaders after train-supported class filtering")
        train_loader, val_loader, test_loader, train_class_stats, val_class_stats = eee_get_dataloaders(args)
        args.dataloader_params['train_class_stats'] = train_class_stats
        args.dataloader_params['val_class_stats'] = val_class_stats
        keysteps = args.dataloader_params['keysteps']
        out_classes = len(keysteps)

    warn_if_split_has_limited_coverage("Train", train_class_stats, list(keysteps.keys()))
    warn_if_split_has_limited_coverage("Validation", val_class_stats, list(keysteps.keys()))
    wandb_logger.config.update(
        {
            "active_classes": list(keysteps.keys()),
            "num_active_classes": out_classes,
            "dropped_train_unseen_classes": dropped_train_classes,
        },
        allow_val_change=True,
    )
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

    if args.wandb_params.get("watch_model", False):
        wandb_logger.watch(model, log="all", log_freq=args.wandb_params.get("watch_log_freq", 100))

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f'./results/job_{cmd_args.job_id}_task_{task}'
    chkpoint_dir = f'./checkpoints/job_{cmd_args.job_id}_task_{task}'

    # print(f"Model: {model}")

    # create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create checkpoint directory if not exists
    if not os.path.exists(chkpoint_dir):
        os.makedirs(chkpoint_dir)

    # Save class-to-new-id mapping for reproducibility/debugging.
    class_mapping = {cls_name: idx for idx, cls_name in enumerate(keysteps.keys())}
    class_mapping_payload = {
        "job_id": cmd_args.job_id,
        "task": task,
        "num_classes": out_classes,
        "class_to_new_id": class_mapping,
        "new_id_to_class": {str(v): k for k, v in class_mapping.items()},
        "dropped_train_unseen_classes": dropped_train_classes,
        "min_train_samples_per_class": args.dataloader_params.get("min_train_samples_per_class", 1),
    }
    class_mapping_filename = f"job_{cmd_args.job_id}_task_{task}_class_mapping.json"
    for out_dir in [results_dir, chkpoint_dir]:
        class_mapping_path = os.path.join(out_dir, class_mapping_filename)
        with open(class_mapping_path, "w") as f:
            json.dump(class_mapping_payload, f, indent=2)
        print(f"Saved class mapping to: {class_mapping_path}")
    
    training_control_params = getattr(args, "training_control_params", {})
    logging_params = getattr(args, "logging_params", {})
    min_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    early_stopping_enabled = training_control_params.get("early_stopping_enabled", True)
    early_stopping_min_delta = training_control_params.get("early_stopping_min_delta", 0.0)
    save_latest_checkpoint = training_control_params.get("save_latest_checkpoint", True)
    evaluate_test_during_training = training_control_params.get("evaluate_test_during_training", False)
    run_final_test = training_control_params.get("run_final_test", True)
    detailed_stdout = logging_params.get("detailed_stdout", False)
    patience = args.learning_params["patience"]

    best_chkpoint_path = os.path.join(chkpoint_dir, "val_best_model.pt")
    latest_chkpoint_path = os.path.join(chkpoint_dir, "last_model.pt")
    training_summary = {
        "job_id": cmd_args.job_id,
        "task": task,
        "best_epoch": None,
        "best_val_loss": None,
        "epochs_completed": 0,
        "early_stopped": False,
        "best_checkpoint_path": best_chkpoint_path,
        "latest_checkpoint_path": latest_chkpoint_path if save_latest_checkpoint else None,
        "train_class_stats": train_class_stats,
        "val_class_stats": val_class_stats,
        "active_classes": list(keysteps.keys()),
        "dropped_train_unseen_classes": dropped_train_classes,
    }

    # # Train the model
    for epoch in range(args.learning_params["epochs"]):
        print("*"*10, "="*10, "*"*10)
        print(f"Epoch: {epoch}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            wandb_logger,
            modality=modality,
            task=task,
            epoch=epoch,
            log_batch_metrics=args.wandb_params.get("log_batch_metrics", False),
            detailed_stdout=detailed_stdout,
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            wandb_logger,
            modality=modality,
            task=task,
            epoch=epoch,
            log_batch_metrics=args.wandb_params.get("log_batch_metrics", False),
            detailed_stdout=detailed_stdout,
        )
        wandb_logger.log({
            "epoch": epoch,
            "train/avg_loss": train_loss,
            "val/loss": val_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
        })
        training_summary["epochs_completed"] = epoch + 1

        if save_latest_checkpoint:
            torch.save(model.state_dict(), latest_chkpoint_path)

        improved = val_loss < (min_val_loss - early_stopping_min_delta)
        if improved:
            min_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            training_summary["best_epoch"] = epoch
            training_summary["best_val_loss"] = val_loss
            wandb_logger.log({"epoch": epoch, "val/best_loss": val_loss})
            torch.save(model.state_dict(), best_chkpoint_path)
            print(f"Saved new best validation checkpoint at epoch {epoch} with val loss {val_loss}")

            if evaluate_test_during_training:
                results = test_model(model, test_loader, criterion, device, wandb_logger, epoch, results_dir, modality=modality, task=task, detailed_stdout=detailed_stdout)
                print(f"[Best Epoch Test] Results: {results}")
        else:
            epochs_without_improvement += 1


        scheduler.step()
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        print("*"*10, "="*10, "*"*10)

        if early_stopping_enabled and epochs_without_improvement >= patience:
            training_summary["early_stopped"] = True
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without validation improvement")
            break

    if best_epoch >= 0 and os.path.exists(best_chkpoint_path):
        print(f"Loading best validation checkpoint from epoch {best_epoch}: {best_chkpoint_path}")
        model.load_state_dict(torch.load(best_chkpoint_path, map_location=device), strict=False)
    else:
        print("[Warning] No best validation checkpoint was saved; using the current model state")

    if run_final_test:
        final_test_epoch = best_epoch if best_epoch >= 0 else training_summary["epochs_completed"] - 1
        results = test_model(model, test_loader, criterion, device, wandb_logger, final_test_epoch, results_dir, modality=modality, task=task, detailed_stdout=detailed_stdout)
        training_summary["final_test_results"] = results
        print(f"Final test results (best validation model): {results}")

    training_summary_filename = f"job_{cmd_args.job_id}_task_{task}_training_summary.json"
    for out_dir in [results_dir, chkpoint_dir]:
        training_summary_path = os.path.join(out_dir, training_summary_filename)
        with open(training_summary_path, "w") as f:
            json.dump(training_summary, f, indent=2)
        print(f"Saved training summary to: {training_summary_path}")

    wandb_logger.summary["best_epoch"] = best_epoch
    wandb_logger.summary["best_val_loss"] = training_summary.get("best_val_loss")
    wandb_logger.summary["train_class_stats"] = train_class_stats
    wandb_logger.summary["val_class_stats"] = val_class_stats
    if "final_test_results" in training_summary:
        wandb_logger.summary["final_test_results"] = training_summary["final_test_results"]
    wandb_logger.finish()
        
