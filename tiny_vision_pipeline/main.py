import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_vision_pipeline import CONSTS
from tiny_vision_pipeline.datasets.cifar_warpper import CIFAR10Wrapped
from tiny_vision_pipeline.datasets.data_loader import load_datasets
from tiny_vision_pipeline.models.MobileNetV3 import MyDragonModel
from tiny_vision_pipeline.trainer import Trainer
from tiny_vision_pipeline.transfor_config import train_transform, test_transform
from tiny_vision_pipeline.utils.save_utils import save_run_state
from tiny_vision_pipeline.utils.utils import create_split_df, split_val_test

# test merge conflict

def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    if device.type == 'cuda':
        print(f"🧠 GPU name: {torch.cuda.get_device_name(device)}")
        print(f"🧮 Memory available: {torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)} MB")

    consts = CONSTS
    x_train_np, y_train_np, x_test_np, y_test_np = load_datasets()

    split_df = create_split_df(
        x_train_np=x_train_np,
        y_train_np=y_train_np,
        x_test_np=x_test_np,
        y_test_np=y_test_np,
        val_ratio=CONSTS.VAL_RATIO,
        seed=CONSTS.SPLIT_SEED
    )

    # Wrap the full test set
    full_test_dataset = CIFAR10Wrapped(x_test_np, y_test_np, transform=test_transform)
    offset = len(x_train_np)

    val_dataset, test_dataset = split_val_test(split_df,full_test_dataset,offset)

    cifar_train = CIFAR10Wrapped(x_train_np, y_train_np, transform = train_transform)
    train_loader = DataLoader(cifar_train, batch_size=CONSTS.BATCH_SIZE, shuffle=True, num_workers=0)  # after debugginh change to num_workers = os.cpu_count() // 2
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)  # after debugginh change to num_workers = os.cpu_count() // 2
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0) # after debugginh change to num_workers = os.cpu_count() // 2

    # Define your model
    model =  MyDragonModel()
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr =CONSTS.LEARNING_RATE)


    
    criterion = nn.CrossEntropyLoss()

    # make a run_dir to save training parameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(CONSTS.RUN_DIR_BASE, f"{CONSTS.SAVE_PATH}_run_{timestamp}")
    # save train configurations
    save_run_state(
        consts=CONSTS,
        run_dir = run_dir,
        data_split_df=split_df)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion,run_dir,  device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train
    trainer.fit(CONSTS.EPOCHS, run_dir)



if __name__ == '__main__':

    main()