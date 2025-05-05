import os.path

import torch
from tqdm import tqdm
from tiny_vision_pipeline.utils.utils import save_checkpoint

import matplotlib.pyplot as plt
import json
import pandas as pd

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, run_dir, scheduler = None, verbose_lr = False,device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.verbose_lr = verbose_lr
        self.device = device
        self.best_accuracy = 0.0
        self.run_dir = run_dir
        # training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.lr_history = []

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

    def save_train_history(self):
        num_epochs = len(self.train_losses)
        epochs = list(range(1, num_epochs + 1))
        data = {
            'epoch': epochs,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        # Add learning rates per param group
        if hasattr(self, 'lr_history') and self.lr_history:
            lr_transposed = list(zip(*self.lr_history))  # transpose to group-wise lists
            for i, group_lrs in enumerate(lr_transposed):
                data[f'lr_group_{i}'] = group_lrs
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.run_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"📄 All training metrics saved to: {csv_path}")

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(15, 5))

        # Loss Plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Acc')
        plt.plot(epochs, self.val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        # Learning Rate Plot
        if hasattr(self, 'lr_history') and self.lr_history:
            plt.subplot(1, 3, 3)
            lr_history_transposed = list(zip(*self.lr_history))  # for multiple param groups
            for i, lr_list in enumerate(lr_history_transposed):
                plt.plot(epochs, lr_list, label=f'LR Group {i}')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Over Epochs')
            plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, 'train_metrics_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Metrics plot saved to: {plot_path}")

    def _log_learning_rate(self, epoch=None):
        lr_dict = {}
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group.get('lr', None)
            if lr is not None:
                lr_dict[f'lr_group_{i}'] = lr
                if epoch is not None:
                    print(f"[Epoch {epoch}] [Group {i}] Learning Rate: {lr:.6f}")
                else:
                    print(f"[Group {i}] Learning Rate: {lr:.6f}")
        return lr_dict

    def fit(self, epochs, checkpoint_path, start_epoch = 0):
        for epoch in range(start_epoch, start_epoch+epochs):
            print(f"Epoch {epoch - start_epoch + 1}/{epochs} (Global epoch {epoch})")

            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.eval_step()

            if self.scheduler is not None:
                # Scheduler step based on validation loss
                self.scheduler.step(val_loss)
                if self.verbose_lr:
                    self._log_learning_rate()

            # store training matrices
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.lr_history.append([
                group['lr'] for group in self.optimizer.param_groups
            ])

            current_epoch = epoch + 1
            print(f"Epoch {current_epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"Val Loss:  {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Save best checkpoint
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if not checkpoint_path:
                    raise ValueError("⚠️ Cannot save checkpoint: `checkpoint_path` is not set or is empty.")


                save_checkpoint(
                    run_dir=self.run_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=current_epoch,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    val_acc=val_acc,
                    val_loss=val_loss
                )

            # Epoch summary
            print(f"Metrics for epoch {epoch + 1}: "
            f"Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}\n")

        # save plot of training metrics
        self.plot_metrics()
        self.save_train_history()
        # save final metrics in json
        final_metrics = {
            "train_accuracy": self.train_accuracies[-1],
            "train_loss": self.train_losses[-1],
            "val_accuracy": self.val_accuracies[-1],
            "val_loss": self.val_losses[-1]
        }
        with open(os.path.join(self.run_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=4)
