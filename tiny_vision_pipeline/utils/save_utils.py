import os
import json
from tiny_vision_pipeline.utils.utils import safe_serialize_const_dict


def save_run_state(consts,  run_dir, data_split_df=None):
    # Create timestamped folder

    os.makedirs(run_dir, exist_ok=True)

    # Save CONSTS
    const_dict = safe_serialize_const_dict(consts)
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(const_dict, f, indent=4)

    # Optionally save split file
    if data_split_df is not None:
        split_path = os.path.join(run_dir, "data_split.csv")
        data_split_df.to_csv(split_path, index=False)

    print(f"🗃️  Run saved to: {run_dir}")
