import json
import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch.utils.data import random_split
from tiny_vision_pipeline.datasets.data_loader import load_datasets
from tiny_vision_pipeline.datasets.cifar_warpper import CIFAR10Wrapped
from datetime import datetime
import os
import shutil

def create_split_df(x_train_np, y_train_np, x_test_np, y_test_np, val_ratio=0.8, seed=42):
    # Train part
    num_train = len(x_train_np)
    train_labels = y_train_np.flatten()
    train_group = np.array(['train'] * num_train)

    # Validation + test split from test set
    num_test = len(x_test_np)
    val_size = int(val_ratio * num_test)
    test_size = num_test - val_size
    generator = torch.Generator().manual_seed(seed)

    val_indices, test_indices = random_split(range(num_test), [val_size, test_size], generator=generator)
    val_indices = set(val_indices.indices)
    test_indices = set(test_indices.indices)

    test_group = np.array(['val' if i in val_indices else 'test' for i in range(num_test)])
    test_labels = y_test_np.flatten()

    # Combine all parts
    all_data = np.concatenate([x_train_np, x_test_np], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_splits = np.concatenate([train_group, test_group], axis=0)

    split_df = pd.DataFrame({
        "index": np.arange(len(all_data)),
        "label": all_labels,
        "split": all_splits
    })

    return split_df


def split_val_test(split_df, full_test_dataset, offset):
    # Use this:
    val_indices = split_df[split_df["split"] == "val"]["index"].to_numpy()
    test_indices = split_df[split_df["split"] == "test"]["index"].to_numpy()

    val_indices_local = val_indices - offset
    test_indices_local = test_indices - offset

    # Extract data and targets for val
    val_data = full_test_dataset.data[val_indices_local]
    val_targets = [full_test_dataset.targets[i] for i in val_indices_local]
    val_transform = getattr(full_test_dataset, 'transform', None)

    # Extract data and targets for test
    test_data = full_test_dataset.data[test_indices_local]
    test_targets = [full_test_dataset.targets[i] for i in test_indices_local]
    test_transform = getattr(full_test_dataset, 'transform', None)

    # Wrap them
    val_dataset = CIFAR10Wrapped(val_data, val_targets, transform=val_transform)
    test_dataset = CIFAR10Wrapped(test_data, test_targets, transform=test_transform)


    return val_dataset, test_dataset


def safe_serialize_const_dict(consts):
    serializable = {}
    for k, v in consts.__dict__.items():
        if k.startswith("__"):
            continue
        try:
            json.dumps(v)  # test if it's serializable
            serializable[k] = v
        except (TypeError, OverflowError):
            serializable[k] = str(v)  # fallback: save as string
    return serializable


def load_split_dataset(run_dir, split_name, transform=None):
    # Load split DataFrame
    split_path = os.path.join(run_dir, "data_split.csv")
    split_df = pd.read_csv(split_path)

    # Load test set only (val/test were split from this)
    _, _, x_test_np, y_test_np = load_datasets()

    # Get indices from the saved full dataset
    offset = len(split_df[split_df["split"] == "train"])
    all_test_indices = np.arange(len(x_test_np))
    df_test_part = split_df.iloc[offset:].reset_index(drop=True)

    # Sanity check: Should align
    assert len(df_test_part) == len(x_test_np), "Test data size mismatch with split file!"

    # Get the local indices for val or test
    split_mask = df_test_part["split"] == split_name
    subset_indices = df_test_part[split_mask].index.to_numpy()  # local index relative to x_test_np

    # Create the dataset and subset
    full_test_dataset = CIFAR10Wrapped(x_test_np, y_test_np, transform=transform)
    return Subset(full_test_dataset, subset_indices)


def log_metrics_dynamic(master_log_path, row_data):
    # Load existing log if it exists
    if os.path.exists(master_log_path):
        df = pd.read_csv(master_log_path)
    else:
        df = pd.DataFrame()

    # Append the new row
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save back to CSV (UTF-8 encoding to avoid weird characters)
    df.to_csv(master_log_path, index=False, encoding="utf-8")

def get_small_dataset(train_dataset, val_dataset):
    small_train_size = int(len(train_dataset) * 0.1)
    small_val_size = int(len(val_dataset) * 0.1)

    # Slice the underlying data and targets
    small_train_data = train_dataset.data[:small_train_size]
    small_train_targets = train_dataset.targets[:small_train_size]

    small_val_data = val_dataset.data[:small_val_size]
    small_val_targets = val_dataset.targets[:small_val_size]

    # Create new wrapped dataset instances
    small_train_dataset = CIFAR10Wrapped(small_train_data, small_train_targets, transform=train_dataset.transform)
    small_val_dataset = CIFAR10Wrapped(small_val_data, small_val_targets, transform=val_dataset.transform)

    return small_train_dataset, small_val_dataset

def get_optimizer(model,learning_rate, weight_decay):
    if weight_decay is None:
        optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def merge_configs(sweep_config, cli_args):
    merged = sweep_config.copy()
    for key, value in vars(cli_args).items():
        # Only update if CLI arg is not None
        if value is not None:
            merged[key.upper()] = value
    return merged


def get_scheduler(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6):
    """
    Creates a ReduceLROnPlateau scheduler.

    Args:
        optimizer: The optimizer to adjust.
        mode: 'min' or 'max' depending on whether you are minimizing or maximizing the monitored quantity.
        factor: How much to reduce the learning rate by. new_lr = old_lr * factor
        patience: Number of epochs with no improvement after which learning rate will be reduced.
        min_lr: Lower bound on the learning rate.

    Returns:
        A learning rate scheduler.
    """
    # Normalize inputs: accept either values or tuples
    mode = mode[0] if isinstance(mode, tuple) else mode
    factor = factor[0] if isinstance(factor, tuple) else factor
    patience = patience[0] if isinstance(patience, tuple) else patience
    min_lr = min_lr[0] if isinstance(min_lr, tuple) else min_lr

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )

    return scheduler

def save_checkpoint(run_dir, model, optimizer, scheduler, epoch, train_acc, train_loss, val_acc, val_loss, extra_info=None):
    """
    Save model checkpoint with flexible metadata.

    Parameters:
        run_dir (str): Directory to save the checkpoint in.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used.
        scheduler (optional): The learning rate scheduler used.
        epoch (int): Current epoch number.
        train_acc (float): Training accuracy.
        train_loss (float): Training loss.
        val_acc (float): Validation accuracy.
        val_loss (float): Validation loss.
        extra_info (dict, optional): Additional items to include in the checkpoint.
    """
    filename = f"train_acc_{train_acc:.2f}_train_loss_{train_loss:.2f}_val_acc_{val_acc:.2f}_val_loss_{val_loss:.2f}.pt"
    full_path = os.path.join(run_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }

    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, full_path)
    print(f"🧪 Best model saved: {full_path}")

def make_new_run_dir(chekpoint_path):
    original_run_dir = os.path.dirname(chekpoint_path)
    original_base = os.path.basename(original_run_dir)
    base_path = os.path.dirname(original_run_dir)
    if "_run_" in original_base:
        base_name = original_base.split("_run_")[0]
    else:
        base_name = original_base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_save_path= f"{base_name}_resume_{timestamp}"
    new_run_dir = os.path.join(base_path,new_save_path)
    os.makedirs(new_run_dir, exist_ok=True)
    print(f"✅ Created new folder: {new_run_dir}")

    files_to_copy = ["train_config.json", "data_split.csv"]
    for filename in files_to_copy:
        src_path = os.path.join(original_run_dir, filename)
        dst_path = os.path.join(new_run_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"📁 Copied {filename}")
        else:
            print(f"⚠️ Warning: {filename} not found in {original_run_dir}")
    return new_run_dir


