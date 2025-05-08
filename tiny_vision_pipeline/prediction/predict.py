import argparse
import json
import os.path
import random

import torch

from tiny_vision_pipeline.models.MobileNetV3 import DragonModel
from tiny_vision_pipeline.prediction.custom_dataset import CustomImageDataset
from tiny_vision_pipeline.transfor_config import get_transform
from tiny_vision_pipeline.utils.convert_to_dict import DotDict
from tiny_vision_pipeline.utils.utils import load_split_dataset
from tiny_vision_pipeline.utils.vizualization_utils import plot_predictions

def predict(args):
    model_dir = os.path.dirname(args.model_path)

    # load model configs
    with open(fr'{model_dir}\train_config.json', 'r') as f:
        train_configs = json.load(f)
    configs = DotDict(train_configs)

    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DragonModel(model_name=configs.MODEL, num_classes=len(configs.CLASSES))
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.test and args.custom_test:
        print("Error: Please provide only one of --test or --custom_test.")
        return


    if args.test:
        # load test dataset
        test_pipeline = get_transform(configs, is_training=False)
        test_dataset = load_split_dataset(model_dir, "test", transform=test_pipeline)

        # select 9 random samples far prediction
        indices = random.sample(range(len(test_dataset)), min(9, len(test_dataset)))
        samples = [test_dataset[i] for i in indices]

    elif os.path.exists(args.custom_test):
    # load custom dataset
        custom_dataset = CustomImageDataset(args.custom_test, configs)

        if len(custom_dataset) == 0:
            print(f"No valid images found in {args.custom_test}")
            return

        # Select 9 random samples
        indices = random.sample(range(len(custom_dataset)), min(9, len(custom_dataset)))
        samples = [custom_dataset[i] for i in indices]

    else:
        print("No valid path was provided.")
        return

    # Make predictions
    images, file_names = zip(*samples)
    images_tensor = torch.stack(images)

    with torch.no_grad():
        outputs = model(images_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
    plot_predictions(images_tensor, predictions, confidences,
                     configs.CLASSES, model_dir)




parser = argparse.ArgumentParser(description="Test a model")
parser.add_argument('--test', action='store_true', default=None,
                    help="test the trained model on the CIFAR test dataset")
parser.add_argument('--custom_test', type=str, default=None,
                    help="Test the trained model on custom images from a file path")
parser.add_argument('--model_path', type=str,
                    default=  r".\tiny_vision_pipeline\trained_models\train_acc_0.95_train_loss_0.17_val_acc_0.86_val_loss_0.52\train_acc_0.95_train_loss_0.17_val_acc_0.86_val_loss_0.52.pt",
                    help="Path to the trained model checkpoint (.pt file)")
args = parser.parse_args()
#trained_model_path = r"C:\\Users\\Gamer\\PycharmProjects\\tiny_vision_pipeline\\tiny_vision_pipeline\\trained_models\\train_acc_0.95_train_loss_0.17_val_acc_0.86_val_loss_0.52\\train_acc_0.95_train_loss_0.17_val_acc_0.86_val_loss_0.52.pt"
predict(args)
