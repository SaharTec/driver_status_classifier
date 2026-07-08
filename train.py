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
from model import DrowsinessLSTM    # noqa: E402


def compute_class_weights(loader, device, num_classes=NUM_CLASSES):
    """Inverse-frequency weights so rare classes count as much as the common
    ones. Returns a (num_classes,) tensor for the loss. `num_classes` reflects
    the kept classes when some are dropped via --exclude."""
    ds = loader.dataset
    labels = [ds.video_labels[v_idx] for v_idx, _ in ds.samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    """Run one full pass over the data loader (train or eval).

    In train mode: computes loss, back-propagates, and updates weights.
    In eval mode:  computes loss and accuracy with gradients disabled.
    Returns (avg_loss, accuracy) over the entire loader.
    """
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
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of stacked LSTM layers (default 2)")
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
    parser.add_argument("--binary", action="store_true",
                        help="collapse to two classes: Alert (absorbs Singing) vs "
                             "Drowsy (absorbs Yawning); Sleeping is excluded")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Class setup -----------------------------------------------------
    # --binary collapses 5 classes → 2: Alert (absorbs Singing) and
    # Drowsy (absorbs Yawning), with Sleeping excluded entirely.
    # This simplifies the decision boundary and improves accuracy when the
    # goal is just "is the driver OK or not?".
    if args.binary:
        merge_arg = [("Singing", "Alert"), ("Yawning", "Drowsy")]
        exclude_arg = list(args.exclude or []) + ["Sleeping"]
    else:
        merge_arg = tuple(args.merge) if args.merge else None
        exclude_arg = args.exclude

    # --- Dataset ---------------------------------------------------------
    # Loads all .npy feature files, applies merge/exclude, builds
    # sliding-window samples, and splits at the video level to avoid leakage.
    train_loader, val_loader, class_names = make_dataloaders(
        batch_size=args.batch_size, stride=args.stride,
        exclude=exclude_arg, merge=merge_arg)
    num_classes = len(class_names)

    # --- Model + optimiser -----------------------------------------------
    model = DrowsinessLSTM(hidden_size=args.hidden_size,
                           num_layers=args.num_layers,
                           num_classes=num_classes).to(device)
    print("Architecture: LSTM")

    # Class weights make rare classes as influential as common ones during
    # training, preventing the model from ignoring minority classes.
    class_weights = compute_class_weights(train_loader, device, num_classes)
    print("Class weights:", [round(w, 2) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # LR scheduler halves the learning rate when val_loss plateaus for 3 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # --- Training loop ---------------------------------------------------
    # We track val_loss (not train_loss) because it reflects how well the
    # model generalises.  The checkpoint with the lowest val_loss is kept;
    # training stops early when val_loss has not improved for `patience` epochs.
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

        # --- Checkpoint + early stopping ---------------------------------
        # Save whenever val_loss improves; stop when it hasn't for `patience`
        # epochs in a row (keeps the best-generalising snapshot, not the last).
        if current_loss < best_val_loss - 1e-4:
            best_val_loss = current_loss
            epochs_no_improve = 0
            ckpt = {"model_type": "lstm", "class_names": class_names,
                    "num_features": NUM_FEATURES,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "state_dict": model.state_dict()}
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
