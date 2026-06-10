"""
Stage C - training loop.

Trains DrowsinessLSTM on the windowed dataset, tracks loss/accuracy, saves the
best weights to models/best_model.pth and plots the learning curves.

Run from the project root:
    python train.py --epochs 30 --batch-size 32 --lr 0.001
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# make the modules in src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import MODELS_DIR                      # noqa: E402
from dataset import make_dataloaders               # noqa: E402
from model import DrowsinessLSTM                    # noqa: E402


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders(
        batch_size=args.batch_size, stride=args.stride)

    model = DrowsinessLSTM(hidden_size=args.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, train=True)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        msg = f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f}"

        if val_loader is not None:
            va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                        optimizer, device, train=False)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            msg += f" | val loss {va_loss:.4f} acc {va_acc:.3f}"
            track = va_acc
        else:
            track = tr_acc

        print(msg)

        if track > best_val:
            best_val = track
            torch.save(model.state_dict(), best_path)

    print(f"\nBest accuracy: {best_val:.3f}  ->  saved to {best_path}")
    plot_history(history)


def plot_history(history):
    has_val = len(history["val_loss"]) > 0
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    if has_val:
        plt.plot(history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train")
    if has_val:
        plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_curves.png", dpi=120)
    print(f"Saved learning curves to {MODELS_DIR / 'training_curves.png'}")
    plt.show()


if __name__ == "__main__":
    main()
