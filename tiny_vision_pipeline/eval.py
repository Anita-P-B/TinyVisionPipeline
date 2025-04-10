import torch
from torch.utils.data import DataLoader
import os
from tiny_vision_pipeline.CONSTS import CONSTS
from tiny_vision_pipeline.evaluate import evaluate_model
from tiny_vision_pipeline.models.MobileNetV3 import MyDragonModel
from tiny_vision_pipeline.transfor_config import test_transform
from tiny_vision_pipeline.utils.utils import load_split_dataset
import argparse

parser = argparse.ArgumentParser(description="evaluate a trained model.")
parser.add_argument('--eval_model_path', type=str, default="default",
                    help="Name for the trained model path.")
args = parser.parse_args()

def evaluate_model_main(run_dir):
    # Load data
    test_dataset = load_split_dataset(run_dir, "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONSTS.BATCH_SIZE)

    # Load model
    model = MyDragonModel(num_classes=len(CONSTS.CLASSES))
    #model_path = os.path.join(run_dir, CONSTS.LOAD_MODEL)
    model.load_state_dict(torch.load(args.eval_model_path))

    model.eval()

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(model, test_loader, run_dir = run_dir, class_names= CONSTS.CLASSES,
                   device=device, plot_examples= True, num_images= 9)


if __name__ == '__main__':
    run_dir = os.path.dirname(args.eval_model_path)
    evaluate_model_main(run_dir)
