import json
import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import random

from collections import defaultdict
import numpy as np


def evaluate_model(model, dataloader, run_dir, class_names=None, device='cpu',
                   plot_examples=False, num_images=9):
    model.to(device)
    model.eval()  # important for proper inference mode

    correct = 0
    total = 0

    all_images = []
    all_preds = []
    all_labels = []
    all_confidences = []

    dataloader_tqdm = tqdm(dataloader, desc="Evaluating Model", leave=False)

    with torch.no_grad():  # no gradient computation needed during evaluation
        for images, labels in dataloader_tqdm:
            images = images.to(device)
            labels = labels.to(device).long()  # ensure labels are int64

            outputs = model(images)  # forward pass
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            all_confidences.extend(confidences.cpu())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            acc_so_far = correct / total
            dataloader_tqdm.set_postfix({"Accuracy": f"{acc_so_far:.4f}"})

            if plot_examples:
                all_images.extend(images.cpu())
                all_preds.extend(predictions.cpu())
                all_labels.extend(labels.cpu())

    accuracy = correct / total
    print(f"🐉 Loaded model test accuracy: {accuracy:.4f}")

    def compute_average_confidence_per_class(all_preds, all_labels, all_confidences, class_names):
        class_conf_dict = defaultdict(list)
        avg_conf_dict = {}

        for pred, label, conf in zip(all_preds, all_labels, all_confidences):
            class_conf_dict[pred.item()].append(conf.item())

        print("📈 Average confidence per predicted class:")
        for class_idx in range(len(class_names)):
            scores = class_conf_dict[class_idx]
            avg_conf = np.mean(scores) if scores else 0.0
            avg_conf_dict[class_names[class_idx]] = round(avg_conf, 4)
            print(f"  {class_names[class_idx]:<12}: {avg_conf:.4f}")
        return avg_conf_dict

    avg_conf_dict = compute_average_confidence_per_class(
        all_preds, all_labels, all_confidences, class_names
    )

    val_path = os.path.join(run_dir, "evaluation.json")
    with open(val_path, "w") as f:
        json.dump({"test_accuracy": accuracy,
                   "average_confidence_per_class": avg_conf_dict
                   }, f, indent=4)
    print(f"📝 Evaluation results saved to: {val_path}")

    # Plot example predictions (randomly chosen)
    if plot_examples and class_names is not None:

        correct_indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p == t]
        incorrect_indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]

        def plot_sample(indices, title, filename):
            if not indices:
                print(f"⚠️ No {title.lower()} predictions to display.")
                return

            chosen = random.sample(indices, k=min(num_images, len(indices)))
            plt.figure(figsize=(12, 8))
            for i, idx in enumerate(chosen):
                img = all_images[idx]
                true = all_labels[idx].item()
                pred = all_preds[idx].item()

                plt.subplot(3, 3, i + 1)
                img_disp = img.permute(1, 2, 0).numpy()
                img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

                plt.imshow(img_disp)
                conf = all_confidences[idx].item() * 100
                plt.title(f"True: {class_names[true]}\nPred: {class_names[pred]} (conf:{conf:.1f}%)")
                plt.axis('off')

            plt.suptitle(f"{title} Predictions", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(run_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"🖼️ {title} predictions plot saved to: {save_path}")

        # plot and save True and False predictions
        plot_sample(correct_indices, title="Correct", filename="correct_predictions.png")
        plot_sample(incorrect_indices, title="Incorrect", filename="incorrect_predictions.png")

    return accuracy
