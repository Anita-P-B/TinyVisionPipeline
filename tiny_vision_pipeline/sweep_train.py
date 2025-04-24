import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
from tiny_vision_pipeline.utils.utils import log_metrics_dynamic
import tiny_vision_pipeline
from tiny_vision_pipeline.CONSTS import Config
from tiny_vision_pipeline.main import main as run_training

parser = argparse.ArgumentParser(description="Run a hyperparameter sweep.")
parser.add_argument('--sweep_name', type=str, default="default",
                    help="Name for the sweep run folder.")
args = parser.parse_args()

# Compose sweep run directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
sweep_run_dir = os.path.join("sweeps", f"sweep_{args.sweep_name}_{timestamp}")
os.makedirs(sweep_run_dir, exist_ok=True)

# Prepare the sweep log path
master_log_path = os.path.join(sweep_run_dir, "all_sweep_results.csv")
with open(master_log_path, mode="w", newline="") as log_file:
    # Define sweep options
    sweep_configs = [
        {"LEARNING_RATE": 0.001},
        { "LEARNING_RATE": 0.001, "BATCH_SIZE": 64 },
        {  "LEARNING_RATE": 0.0005, "BATCH_SIZE": 32 },
        {"LEARNING_RATE": 0.0005, "BATCH_SIZE": 64}
    ]


    for i, config in enumerate(sweep_configs):
        print(f"\n🌀 Starting sweep {i + 1}/{len(sweep_configs)} with config: {config}")

        # Create default for comparison
        default_config = Config()
        consts = Config()

        consts.SWEEP_MODE = True

        consts.update_from_dict(config)

        # Find changed keys
        changed = []
        for key, value in config.items():
            if hasattr(default_config, key):
                if getattr(default_config, key) != value:
                    changed.append(f"{key.lower()}{value}")

        # Create dynamic SAVE_PATH
        if changed:
            save_name = "sweep_" + "_".join(changed)
        else:
            from datetime import datetime

            save_name = "sweep_default_" + datetime.now().strftime("%Y%m%d_%H%M")

        consts.SAVE_PATH = save_name

        # 👇 Set run_dir inside the sweep folder
        consts.RUN_DIR_BASE = sweep_run_dir

        # Register CONSTS globally

        tiny_vision_pipeline.CONSTS = consts

        # Train
        run_training(consts=consts)

        # Locate the latest run dir
        matching_dirs = [d for d in os.listdir(sweep_run_dir) if d.startswith(consts.SAVE_PATH)]
        latest_run = sorted(matching_dirs)[-1]
        latest_run_dir = os.path.join(sweep_run_dir, latest_run)

        # Load final metrics
        metrics_path = os.path.join(latest_run_dir, "final_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {"train_accuracy": None, "train_loss": None,
                       "val_accuracy": None, "val_loss": None}

        row_data = {
            "run_dir": latest_run_dir,
            **config,
            **metrics
        }
        log_metrics_dynamic(master_log_path, row_data)

print(f"\n📜 Sweep complete! Results saved to: {master_log_path}")
