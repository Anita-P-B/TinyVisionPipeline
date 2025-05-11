import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from tiny_vision_pipeline.CONSTS import CONSTS
from tiny_vision_pipeline.evaluate.eval_logger import ModelEvaluator
from tiny_vision_pipeline.models.MobileNetV3 import DragonModel
from tiny_vision_pipeline.transfor_config import get_transform
from tiny_vision_pipeline.utils.convert_to_dict import DotDict
from tiny_vision_pipeline.utils.utils import load_split_dataset

parser = argparse.ArgumentParser(description="Evaluate a trained model.")
parser.add_argument('--eval_model_path', type=str, default=None,
                    help="Path to the trained model checkpoint.")
args = parser.parse_args()

eval_model_path = args.eval_model_path or CONSTS.LOAD_MODEL


def evaluate_model_main(run_dir, eval_model_path):
    # Load data

    with open(os.path.join(run_dir, 'train_config.json'), 'r') as f:
        train_configs = json.load(f)
    consts = DotDict(train_configs)

    test_pipeline = get_transform(consts, is_training=False)

    test_dataset = load_split_dataset(run_dir, "test", transform=test_pipeline)
    test_loader = DataLoader(test_dataset, batch_size=consts.BATCH_SIZE)


    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DragonModel(model_name=consts.MODEL, num_classes=len(consts.CLASSES))
    checkpoint = torch.load(eval_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Evaluate
    evaluator = ModelEvaluator(model, dataloader=test_loader, run_dir=run_dir,
                               class_names=consts.CLASSES, device=device, plot_examples=True)
    evaluator.evaluate_model()
    evaluator.save_results()
    evaluator.plot_examples_grid()


if __name__ == '__main__':
    run_dir = os.path.dirname(eval_model_path)
    evaluate_model_main(run_dir, eval_model_path)
