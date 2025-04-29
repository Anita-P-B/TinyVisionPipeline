import os.path

import torch
from tqdm import tqdm


import matplotlib.pyplot as plt
import json

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, run_dir, scheduler = None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.best_accuracy = 0.0
        self.run_dir = run_dir
        # training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_step(self):
        self.model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_loader_tqdm = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
            images = images.to(self.device)
            labels = labels.to(self.device).long()


            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Update tqdm
            train_loader_tqdm.set_postfix({
                "Batch": f"{batch_idx + 1}/{len(self.train_loader)}",
                "Loss": f"{loss.item():.4f}"
            })

        # Compute average metrics
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total  # 0.0 - 1.0
        return avg_train_loss, train_acc

    def eval_step(self):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_loader_tqdm = tqdm(self.val_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # update tqdm
                val_loader_tqdm.set_postfix({
                    "Loss": f"{loss.item():.4f}"
                })

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        return avg_val_loss, val_acc

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Acc')
        plt.plot(epochs, self.val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()

        plot_path = os.path.join(self.run_dir, 'train_metrics_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Metrics plot saved to: {plot_path}")

    def fit(self, epochs, checkpoint_path=None, start_epoch = 0):
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.eval_step()

            if self.scheduler is not None:
                # Scheduler step based on validation loss
                self.scheduler.step(val_loss)

            # store training matrices
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            current_epoch = epoch + 1
            print(f"Epoch {current_epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"Val Loss:  {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Save best checkpoint
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if checkpoint_path:

                    formatted_path = f"train_acc_{train_acc:.2f}_train_loss_{train_loss:.2f}_val_acc_{val_acc:.2f}val_loss_{val_loss:.2f}.pt"
                    full_path = os.path.join(self.run_dir, formatted_path)
                    torch.save({
                        'epoch': current_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                    },  full_path)
                    print(f"🧪 Best model saved: {full_path}")

            # Epoch summary
            print(f"Metrics for epoch {epoch + 1}: "
            f"Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}\n")

        # save plot of training metrics
        self.plot_metrics()
        # save final metrics in json
        final_metrics = {
            "train_accuracy": self.train_accuracies[-1],
            "train_loss": self.train_losses[-1],
            "val_accuracy": self.val_accuracies[-1],
            "val_loss": self.val_losses[-1]
        }
        with open(os.path.join(self.run_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=4)
