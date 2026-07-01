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

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# make the modules in src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import MODELS_DIR, NUM_CLASSES, NUM_FEATURES  # noqa: E402
from dataset import make_dataloaders               # noqa: E402
from model import DrowsinessLSTM, DrowsinessTCN    # noqa: E402


# --- previous version (hard-wired to NUM_CLASSES) ------------------------
# def compute_class_weights(loader, device):
#     """Inverse-frequency weights so rare classes (e.g. Sleeping) count
#     as much as the common ones. Returns a (NUM_CLASSES,) tensor for the loss."""
#     ds = loader.dataset
#     labels = [ds.video_labels[v_idx] for v_idx, _ in ds.samples]
#     counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
#     counts = np.clip(counts, 1.0, None)            # avoid div-by-zero
#     weights = counts.sum() / (NUM_CLASSES * counts)
#     return torch.tensor(weights, dtype=torch.float32, device=device)
# -------------------------------------------------------------------------
def compute_class_weights(loader, device, num_classes=NUM_CLASSES):
    """Inverse-frequency weights so rare classes count as much as the common
    ones. Returns a (num_classes,) tensor for the loss. `num_classes` reflects
    the kept classes when some are dropped via --exclude."""
    ds = loader.dataset
    labels = [ds.video_labels[v_idx] for v_idx, _ in ds.samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)            # avoid div-by-zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
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

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["lstm", "tcn"], default="lstm",
                        help="model architecture: lstm (default) or tcn")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of stacked LSTM layers (default 2, lstm only)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patience", type=int, default=7,
                        help="stop after this many epochs without val_loss improvement")
    parser.add_argument("--exclude", nargs="+", default=None,
                        metavar="CLASS",
                        help="class names whose videos are dropped from training, "
                             "e.g. --exclude Sleeping")
    parser.add_argument("--merge", nargs=2, default=None,
                        metavar=("SRC", "DST"),
                        help="merge SRC class into DST, e.g. --merge Sleeping Drowsy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- previous version (no exclude) ----------------------------------
    # train_loader, val_loader = make_dataloaders(
    #     batch_size=args.batch_size, stride=args.stride)
    #
    # model = DrowsinessLSTM(hidden_size=args.hidden_size).to(device)
    # class_weights = compute_class_weights(train_loader, device)
    # --------------------------------------------------------------------
    train_loader, val_loader, class_names = make_dataloaders(
        batch_size=args.batch_size, stride=args.stride,
        exclude=args.exclude, merge=tuple(args.merge) if args.merge else None)
    num_classes = len(class_names)

    if args.model == "tcn":
        model = DrowsinessTCN(num_classes=num_classes).to(device)
    else:
        model = DrowsinessLSTM(hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               num_classes=num_classes).to(device)
    print(f"Architecture: {args.model.upper()}")
    class_weights = compute_class_weights(train_loader, device, num_classes)
    print("Class weights:", [round(w, 2) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Early stopping tracks val_loss (the signal that actually shows overfitting).
    best_val_loss = float("inf")
    epochs_no_improve = 0

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
            current_loss = va_loss
        else:
            current_loss = tr_loss

        print(msg)
        scheduler.step(current_loss)

        # Save the model with the lowest val_loss, and stop once it stops
        # improving for `patience` epochs (so we keep the best-generalizing
        # epoch instead of the over-fitted last one).
        if current_loss < best_val_loss - 1e-4:
            best_val_loss = current_loss
            epochs_no_improve = 0
            ckpt = {"model_type": args.model, "class_names": class_names,
                    "num_features": NUM_FEATURES,
                    "state_dict": model.state_dict()}
            if args.model == "lstm":
                ckpt.update({"hidden_size": args.hidden_size,
                             "num_layers": args.num_layers})
            torch.save(ckpt, best_path)
            print(f"        saved (best val loss {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val_loss improvement for {args.patience} epochs).")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}  ->  saved to {best_path}")
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
