import argparse
import os
from datetime import datetime
from idlelib.colorizer import color_config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_vision_pipeline.CONSTS import CONSTS
from tiny_vision_pipeline.datasets.cifar_warpper import CIFAR10Wrapped
from tiny_vision_pipeline.datasets.data_loader import load_datasets
from tiny_vision_pipeline.models.MobileNetV3 import DragonModel
from tiny_vision_pipeline.trainer import Trainer
from tiny_vision_pipeline.transfor_config import get_transform
from tiny_vision_pipeline.utils.save_utils import save_run_state
from tiny_vision_pipeline.utils.utils import create_split_df, split_val_test, get_small_dataset
from tiny_vision_pipeline.utils.utils import get_optimizer, get_scheduler

def main(consts=None, user_config=None):
    consts = consts or CONSTS

    # Apply overrides if provided (manual run)
    # If user_config is a dictionary of overrides
    if user_config:
        for key, value in user_config.items():
            if hasattr(consts, key):
                setattr(consts, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    if device.type == 'cuda':
        print(f"🧠 GPU name: {torch.cuda.get_device_name(device)}")
        print(f"🧮 Memory available: {torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)} MB")

    x_train_np, y_train_np, x_test_np, y_test_np = load_datasets()

    split_df = create_split_df(
        x_train_np=x_train_np,
        y_train_np=y_train_np,
        x_test_np=x_test_np,
        y_test_np=y_test_np,
        val_ratio=consts.VAL_RATIO,
        seed=consts.SPLIT_SEED
    )

    # define train and test transform pipelines
    train_pipeline = get_transform(consts, is_training=True)
    val_pipeline = get_transform(consts, is_training=False)

    # Wrap the full test set
    full_test_dataset = CIFAR10Wrapped(x_test_np, y_test_np, transform=val_pipeline)
    offset = len(x_train_np)

    cifar_val, cifar_test = split_val_test(split_df, full_test_dataset, offset)
    cifar_train = CIFAR10Wrapped(x_train_np, y_train_np, transform=train_pipeline)

    if consts.SMALL_DATASET:
        cifar_train, cifar_val = get_small_dataset(cifar_train, cifar_val)

    train_loader = DataLoader(cifar_train, batch_size=consts.BATCH_SIZE, shuffle=True,
                              num_workers=0)  # after debugginh change to num_workers = os.cpu_count() // 2
    val_loader = DataLoader(cifar_val, batch_size=32,
                            num_workers=0)  # after debugginh change to num_workers = os.cpu_count() // 2
    test_loader = DataLoader(cifar_test, batch_size=32,
                             num_workers=0)  # after debugginh change to num_workers = os.cpu_count() // 2

    # Define your model
    model = DragonModel(model_name = consts.MODEL, dropout_rate= consts.DROPOUT_RATE)
    # Define optimizer and loss
    optimizer = get_optimizer(model, consts.LEARNING_RATE, consts.WEIGHT_DECAY)
    if consts.SCHEDULER:
        scheduler = get_scheduler(optimizer, mode = consts.MODE,
                                  factor= consts.FACTOR, patience= consts.PATIENCE,
                                  min_lr= consts.MIN_LR)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if not consts.SWEEP_MODE and consts.CHECKPOINT_PATH and os.path.isfile(consts.CHECKPOINT_PATH):
        print(f"🧙‍♂️ Loading checkpoint from {consts.CHECKPOINT_PATH}")
        checkpoint = torch.load(consts.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to the correct device (CUDA)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if 'scheduler_state_dict' in checkpoint and scheduler is not None and checkpoint[
            'scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        run_dir = os.path.dirname(consts.CHECKPOINT_PATH)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"🔄 Resumed from epoch {start_epoch}")
    else:
        print("🛡️ No checkpoint provided, starting from scratch.")
        start_epoch = 0
        # make a run_dir to save training parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(consts.RUN_DIR_BASE, f"{consts.SAVE_PATH}_run_{timestamp}")
        # save train configurations
        save_run_state(
            consts=consts,
            run_dir=run_dir,
            data_split_df=split_df)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, run_dir, scheduler,
                      device='cuda' if torch.cuda.is_available() else 'cpu')

    print(f"📦 Using {'small' if consts.SMALL_DATASET else 'full'} dataset for this sweep.")
    # Train
    trainer.fit(consts.EPOCHS, run_dir, start_epoch=start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--small_dataset', action='store_true', default=None,
                        help="train on 0.1 portion of the dataset for debug.")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the model.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="resumes training from "
                                                                          "given chekpoint.")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs in train.")
    args = parser.parse_args()

    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}

    main(user_config=user_config)
