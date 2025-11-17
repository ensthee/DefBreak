#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MLP-SAMPLE")


# Dataset
class TextDataset(Dataset):
    def __init__(self, finetuned_dir, unfinetuned_dir):
        self.finetuned = self._collect(finetuned_dir, 1, "finetuned")
        self.unfinetuned = self._collect(unfinetuned_dir, 0, "unfinetuned")

        self.files = [f for f, _ in self.finetuned + self.unfinetuned]
        self.labels = [l for _, l in self.finetuned + self.unfinetuned]

        logger.info(f"Total samples {len(self.files)} (FT={len(self.finetuned)}, UF={len(self.unfinetuned)})")

    def _collect(self, root, label, desc):
        root = Path(root)
        if not root.exists():
            logger.warning(f"{desc} missing: {root}")
            return []

        items = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".txt"):
                    items.append((os.path.join(r, f), label))
        logger.info(f"{desc} loaded: {len(items)} txt")
        return items

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        lab = self.labels[idx]

        try:
            arr = np.loadtxt(fp)
            if arr.ndim != 1:
                raise ValueError
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        return torch.tensor(arr, dtype=torch.float32), torch.tensor(lab)


# Model
class MLPClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.fc(x)


# Evaluate
def evaluate(model, loader, device):
    model.eval()
    labs, preds, scores = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1]
            _, pred = out.max(1)

            labs.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            scores.extend(prob.cpu().numpy())

    labs = np.array(labs)
    preds = np.array(preds)
    scores = np.array(scores)

    acc = accuracy_score(labs, preds)
    pre = precision_score(labs, preds, zero_division=0)
    auc = roc_auc_score(labs, scores)

    return acc, pre, auc, labs, scores


# Train
def train(model, train_loader, val_loader, device, epochs, lr, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    patience = 20
    p = 0

    train_loss_list = []
    val_acc_list = []

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        train_loss_list.append(total_loss / len(train_loader))

        acc, pre, auc, _, _ = evaluate(model, val_loader, device)
        val_acc_list.append(acc * 100)

        logger.info(f"[Epoch {ep+1}] Loss={train_loss_list[-1]:.4f}, ValAcc={acc*100:.2f}%, AUC={auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            p = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved ValAcc={acc*100:.2f}%")
        else:
            p += 1
            if p >= patience:
                logger.info("EarlyStopping")
                break

    return train_loss_list, val_acc_list


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_dir", required=True)
    parser.add_argument("--unfinetuned_dir", required=True)
    parser.add_argument("--model_path", default="best_mlp_enhanced.pth")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ds = TextDataset(args.finetuned_dir, args.unfinetuned_dir)

    dim = len(ds[0][0])
    logger.info(f"Input dim: {dim}")

    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = MLPClassifier(dim).to(device)

    loss_list, acc_list = train(
        model, train_loader, val_loader, device,
        args.epochs, args.lr, args.model_path
    )

    best_model = MLPClassifier(dim).to(device)
    best_model.load_state_dict(torch.load(args.model_path))

    acc, pre, auc, labs, scores = evaluate(best_model, val_loader, device)
    logger.info(f"Final: ACC={acc:.4f}, PRE={pre:.4f}, AUC={auc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title("Train Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(acc_list)
    plt.title("Val Accuracy")
    plt.grid()
    plt.savefig("mlp_enhanced_curves.png")

    try:
        fpr, tpr, _ = roc_curve(labs, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.legend()
        plt.grid()
        plt.title("ROC Curve")
        plt.savefig("mlp_enhanced_roc.png")
    except:
        pass


if __name__ == "__main__":
    main()
