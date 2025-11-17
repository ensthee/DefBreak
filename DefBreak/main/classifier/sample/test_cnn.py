import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from pathlib import Path
from typing import Optional
import logging
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
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
            logger.warning("Finetuned directory not found")
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
            logger.warning("Unfinetuned directory not found")
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
                logger.warning(f"Unexpected data shape in file")
                return None
            return data
        except Exception as e:
            logger.error(f"Error loading file: {e}")
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
            out = self.conv1(x)
            return out.view(1, -1).size(1)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.conv1(x1)
        x1 = x1.view(x1.size(0), -1)

        if x2.dim() == 1:
            x2 = x2.unsqueeze(1)
        x2 = torch.relu(self.fc1(x2))

        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)


# Metrics / Evaluation
def print_metrics(y_true, y_pred, y_prob, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    pos_label = 1
    neg_label = 0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    if cm.shape == (1, 1):
        if y_true[0] == pos_label:
            cm = np.array([[0, 0], [0, cm[0, 0]]])
        else:
            cm = np.array([[cm[0, 0], 0], [0, 0]])
    elif cm.shape != (2, 2):
        cm = np.array([[0, 0], [0, 0]])

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    roc_auc = 0.0
    fpr_roc, tpr_roc = np.array([0, 1]), np.array([0, 1])
    if len(np.unique(y_true)) > 1:
        try:
            fpr_roc, tpr_roc, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr_roc, tpr_roc)
        except:
            roc_auc = 0.5
    else:
        roc_auc = np.nan

    return fpr_roc, tpr_roc, roc_auc, accuracy, precision, recall, f1, specificity


# Main
def main():
    parser = argparse.ArgumentParser(description="Test CNN classifier")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    base_dir = Path(args.test_data)
    test_finetuned_dir = str(base_dir / "finetuned")
    test_unfinetuned_dir = str(base_dir / "unfinetuned")

    test_dataset = TextDataset(test_finetuned_dir, test_unfinetuned_dir)

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN1DClassifier().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="Testing")
        for x1, x2, labels in pbar_test:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model(x1, x2)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_predicted_probs.extend(probabilities[:, 1].cpu().numpy())

    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    all_predicted_probs = np.array(all_predicted_probs)

    class_names = ["Unfinetuned", "Finetuned"]

    fpr_all, tpr_all, auc_all, acc_all, prec_all, rec_all, f1_all, spec_all = (
        print_metrics(all_true_labels, all_predicted_labels, all_predicted_probs, class_names)
    )

    plt.figure(figsize=(8, 6))
    if not np.isnan(auc_all):
        plt.plot(fpr_all, tpr_all, color="darkorange", lw=2, label=f"AUC = {auc_all:.2f}")
    else:
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - CNN")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve_cnn.png")
    plt.close()

    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - CNN")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix_cnn.png")
    plt.close()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
