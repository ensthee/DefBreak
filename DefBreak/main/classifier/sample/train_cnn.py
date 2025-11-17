import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from pathlib import Path
from typing import Optional
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Dataset
class TextDataset(Dataset):
    def __init__(self, finetuned_dir, unfinetuned_dir):
        self.finetuned_files = []
        self.unfinetuned_files = []
        self.all_files = []
        self.labels = []

        try:
            self.finetuned_files = [
                os.path.join(finetuned_dir, f)
                for f in os.listdir(finetuned_dir)
                if f.endswith(".txt")
            ]
            self.labels.extend([1] * len(self.finetuned_files))
            logger.info(f"Found {len(self.finetuned_files)} finetuned files")
        except FileNotFoundError:
            logger.warning(f"Finetuned directory not found")
        except Exception as e:
            logger.error(f"Error reading finetuned directory: {e}")

        try:
            self.unfinetuned_files = [
                os.path.join(unfinetuned_dir, f)
                for f in os.listdir(unfinetuned_dir)
                if f.endswith(".txt")
            ]
            self.labels.extend([0] * len(self.unfinetuned_files))
            logger.info(f"Found {len(self.unfinetuned_files)} unfinetuned files")
        except FileNotFoundError:
            logger.warning(f"Unfinetuned directory not found")
        except Exception as e:
            logger.error(f"Error reading unfinetuned directory: {e}")

        self.all_files = self.finetuned_files + self.unfinetuned_files

        if not self.all_files:
            raise ValueError("No data files found")

        logger.info(f"Total files loaded: {len(self.all_files)}")

    def __len__(self):
        return len(self.all_files)

    def load_single_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            data = np.loadtxt(file_path)
            if data.shape[0] != 120001:
                logger.warning(f"Unexpected data shape in {file_path}")
                return None
            return data
        except Exception as e:
            logger.error(f"Loading error: {e}")
            return None

    def __getitem__(self, idx):
        while True:
            file_path = self.all_files[idx]
            data = self.load_single_file(file_path)

            if data is not None:
                x1 = torch.FloatTensor(data[:120000])
                x2 = torch.FloatTensor(data[120000:])
                label = torch.LongTensor([self.labels[idx]])
                return x1, x2, label.squeeze()
            else:
                idx = (idx + 1) % len(self.all_files)


# Model
class CNN1DClassifier(nn.Module):
    def __init__(self):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.feature_size = self._get_conv_output_size(120000)
        logger.info(f"Conv feature size: {self.feature_size}")

        self.fc1 = nn.Linear(1, 64)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            x = torch.rand(1, 1, shape)
            output = self.conv1(x)
            return output.view(1, -1).size(1)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.conv1(x1)
        x1 = x1.view(x1.size(0), -1)

        if x2.dim() == 1:
            x2 = x2.unsqueeze(1)
        x2 = torch.relu(self.fc1(x2))

        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)


# Training
def train_model(model, train_loader, val_loader, device, num_epochs, lr, model_save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    train_losses = []
    val_accuracies = []

    logger.info("Training started")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (x1, x2, labels) in enumerate(pbar):
            if x1 is None or x2 is None or labels is None:
                continue

            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Acc": f"{100.0 * correct_train / total_train:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f"Train Loss: {epoch_loss:.4f}")

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for x1, x2, labels in pbar_val:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                outputs = model(x1, x2)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                pbar_val.set_postfix({"Val Acc": f"{100.0 * correct_val / total_val:.2f}%"})


        if total_val > 0:
            val_acc = 100.0 * correct_val / total_val
            val_accuracies.append(val_acc)
            logger.info(f"Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"New best model saved ({best_val_acc:.2f}%)")

    logger.info("Training completed")
    return train_losses, val_accuracies


# Main
def main():
    parser = argparse.ArgumentParser(description="Train CNN classifier")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_path", default="best_cnn_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    base_dir = Path(args.train_data)
    train_finetuned_dir = str(base_dir / "finetuned")
    train_unfinetuned_dir = str(base_dir / "unfinetuned")

    if not base_dir.exists():
        logger.error("Training directory missing")
        return

    full_train_dataset = TextDataset(train_finetuned_dir, train_unfinetuned_dir)

    if len(full_train_dataset) == 0:
        return

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    if train_size == 0 or val_size == 0:
        if train_size == 0:
            train_size = len(full_train_dataset)
            val_size = 0

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ) if val_size > 0 else None

    model = CNN1DClassifier().to(device)

    if val_loader:
        train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, device,
            args.epochs, args.learning_rate, args.model_path
        )

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_losses)
        plt.title("Training Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.epochs + 1), val_accuracies)
        plt.title("Validation Accuracy")
        plt.grid(True)

        plt.savefig("cnn_training_curves.png")
        plt.close()

    logger.info(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
