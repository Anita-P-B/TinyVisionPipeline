import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch.utils.data import random_split

from tiny_vision_pipeline.datasets.cifar_warpper import CIFAR10Wrapped
from tiny_vision_pipeline.datasets.data_loader import load_datasets


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

def split_val_test(split_df, offset):
    # Use this:
    val_indices = split_df[split_df["split"] == "val"]["index"].to_numpy()
    test_indices = split_df[split_df["split"] == "test"]["index"].to_numpy()

    val_indices_local = val_indices - offset
    test_indices_local = test_indices - offset

    # val_dataset = Subset(full_test_dataset, val_indices_local)
    # test_dataset = Subset(full_test_dataset, test_indices_local)
    return val_indices_local, test_indices_local


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

def get_all_labels_from_loader(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    return all_labels

def plot_class_distribution(dataset, name="Dataset"):
    labels = get_all_labels_from_loader(dataset)
    counts = Counter(labels)
    keys = list(range(10))  # assuming CIFAR-10
    values = [counts[k] for k in keys]

    plt.bar(keys, values)
    plt.xticks(keys, ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               rotation=45)
    plt.title(f"Class Distribution: {name}")
    plt.savefig(f"../samples/classes_distribution_{name}.png")
    plt.close()
