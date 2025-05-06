import json
import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import random

from collections import defaultdict
import numpy as np

class ModelEvaluator:
    def __init__(self, model, dataloader, run_dir, class_names=None, device='cpu',
                 plot_examples=False, num_images=9):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.run_dir = run_dir
        self.class_names = class_names
        self.device = device
        self.plot_examples = plot_examples
        self.num_images = num_images

        self.correct = 0
        self.total = 0
        self.accuracy = 0.0
        self.all_images = []
        self.all_preds = []
        self.all_labels = []
        self.all_confidences = []
        self.avg_conf_dict = {}
        self.accuracy_dict = {}


    def evaluate_model(self):
        self.model.to(self.device)
        self.model.eval()  # important for proper inference mode

        dataloader_tqdm = tqdm(self.dataloader, desc="Evaluating Model", leave=False)

        with torch.no_grad():  # no gradient computation needed during evaluation
            for images, labels in dataloader_tqdm:
                images = images.to(self.device)
                labels = labels.to(self.device).long()  # ensure labels are int64

                outputs = self.model(images)  # forward pass
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)


                self.correct += (predictions == labels).sum().item()
                self.total += labels.size(0)

                self.all_preds.extend(predictions.cpu())
                self.all_labels.extend(labels.cpu())
                self.all_confidences.extend(confidences.cpu().tolist())

                acc_so_far = self.correct / self.total
                dataloader_tqdm.set_postfix({"Accuracy": f"{acc_so_far:.4f}"})

                if self.plot_examples:
                    self.all_images.extend(images.cpu())


        self.accuracy = self.correct / self.total
        print(f"🐉 Loaded model test accuracy: {self.accuracy:.4f}")

    def compute_average_confidence_per_class(self):
        class_conf_dict = defaultdict(list)

        for pred, label, conf in zip(self.all_preds, self.all_labels, self.all_confidences):
            class_conf_dict[pred.item()].append(conf)

        print("📈 Average confidence per predicted class:")
        for class_idx in range(len(self.class_names)):
            scores = class_conf_dict[class_idx]
            avg_conf = np.mean(scores) if scores else 0.0
            self.avg_conf_dict[self.class_names[class_idx]] = round(avg_conf, 4)
            print(f"  {self.class_names[class_idx]:<12}: {avg_conf:.4f}")

    def compute_accuracy_per_class(self):
        # group accuracy by true class
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for pred, label in zip(self.all_preds, self.all_labels):
            pred = pred.item() if isinstance(pred, torch.Tensor) else pred
            label = label.item() if isinstance(label, torch.Tensor) else label
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        print("📊 Accuracy per true class:")
        for class_idx in range(len(self.class_names)):
            total = class_total[class_idx]
            correct = class_correct[class_idx]
            acc = correct / total if total > 0 else 0.0
            self.accuracy_dict[self.class_names[class_idx]] = round(acc, 4)
            print(f"  {self.class_names[class_idx]:<12}: {acc:.4f}")

    def save_results(self):
        val_path = os.path.join(self.run_dir, "evaluation.json")
        self.compute_accuracy_per_class()
        self.compute_average_confidence_per_class()
        with open(val_path, "w") as f:
            json.dump({"test_accuracy": self.accuracy,
                       "avarage_accuracy_per_class": self.accuracy_dict ,
                   "average_confidence_per_class": self.avg_conf_dict
                   }, f, indent=4)
        print(f"📝 Evaluation results saved to: {val_path}")

    def plot_examples_grid(self):
        # Plot example predictions (randomly chosen)
        if not self.plot_examples or self.class_names is None:
            return

        correct_indices = [i for i, (p, t) in enumerate(zip(self.all_preds, self.all_labels)) if p == t]
        incorrect_indices = [i for i, (p, t) in enumerate(zip(self.all_preds, self.all_labels)) if p != t]

        def plot_sample(indices, title, filename):
            if not indices:
                print(f"⚠️ No {title.lower()} predictions to display.")
                return

            chosen = random.sample(indices, k=min(self.num_images, len(indices)))
            plt.figure(figsize=(12, 8))
            for i, idx in enumerate(chosen):
                img = self.all_images[idx]
                true = self.all_labels[idx]
                pred = self.all_preds[idx]

                plt.subplot(3, 3, i + 1)
                img_disp = img.permute(1, 2, 0).numpy()
                img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

                plt.imshow(img_disp)
                conf = self.all_confidences[idx] * 100
                plt.title(f"True: {self.class_names[true]}\nPred: {self.class_names[pred]} (conf:{conf:.1f}%)")
                plt.axis('off')

            plt.suptitle(f"{title} Predictions", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(self.run_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"🖼️ {title} predictions plot saved to: {save_path}")

        # plot and save True and False predictions
        plot_sample(correct_indices, title="Correct", filename="correct_predictions.png")
        plot_sample(incorrect_indices, title="Incorrect", filename="incorrect_predictions.png")

