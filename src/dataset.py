"""
Stage B - PyTorch Dataset & DataLoader.

Loads the .npy feature files produced by preprocess.py and turns each video
into many fixed-length training samples using a sliding window of
SEQUENCE_LENGTH frames. Every window inherits the label of its source video.

To avoid data leakage, the train/val split is done at the VIDEO level
(windows from one video never appear in both splits).
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- previous version (kept so we can restore the no-exclude setup) ---
# from config import (PROCESSED_DIR, LABELS_FILE, SEQUENCE_LENGTH, NUM_FEATURES)
from config import (PROCESSED_DIR, LABELS_FILE, SEQUENCE_LENGTH, NUM_FEATURES,
                    CLASSES, CLASS_TO_IDX, NUM_CLASSES)


def _load_entries():
    """Return a list of (features_array, label) for every processed video."""
    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"{LABELS_FILE} not found. Run 'python src/preprocess.py' first."
        )
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    entries = []
    for npy_name, label in labels.items():
        arr = np.load(PROCESSED_DIR / npy_name).astype(np.float32)
        if arr.shape[0] >= SEQUENCE_LENGTH:
            entries.append((arr, int(label)))
    return entries


class DrowsinessDataset(Dataset):
    """
    Builds sliding-window samples from a list of (features, label) videos.

    Each item is:
        x : FloatTensor of shape (SEQUENCE_LENGTH, NUM_FEATURES)
        y : LongTensor scalar (the class index)
    """

    def __init__(self, entries, seq_len=SEQUENCE_LENGTH, stride=1):
        self.seq_len = seq_len
        self.samples = []  # list of (video_idx, start_frame)
        self.arrays = [arr for arr, _ in entries]
        self.video_labels = [lbl for _, lbl in entries]

        for v_idx, arr in enumerate(self.arrays):
            last_start = arr.shape[0] - seq_len
            for start in range(0, last_start + 1, stride):
                self.samples.append((v_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_idx, start = self.samples[idx]
        window = self.arrays[v_idx][start:start + self.seq_len]
        x = torch.from_numpy(window)                       # (seq_len, NUM_FEATURES)
        y = torch.tensor(self.video_labels[v_idx], dtype=torch.long)
        return x, y


# --- previous version (no class filtering) -------------------------------
# def make_dataloaders(batch_size=32, val_split=0.2, stride=1, seed=42):
#     """
#     Split videos into train/val, build sliding-window datasets and return
#     (train_loader, val_loader).
#     """
#     entries = _load_entries()
#     if not entries:
#         raise RuntimeError("No usable videos found in data/processed.")
#
#     random.seed(seed)
#     random.shuffle(entries)
# -------------------------------------------------------------------------
def make_dataloaders(batch_size=32, val_split=0.2, stride=1, seed=42, exclude=None):
    """
    Split videos into train/val, build sliding-window datasets and return
    (train_loader, val_loader, class_names).

    `exclude` is an optional list of class names (e.g. ["Sleeping", "Distracted"])
    whose videos are dropped. The remaining labels are remapped to a contiguous
    0..K-1 range so the loss and the model's output layer stay aligned.
    `class_names` is the kept classes in their new index order.
    """
    entries = _load_entries()
    if not entries:
        raise RuntimeError("No usable videos found in data/processed.")

    if exclude:
        unknown = [c for c in exclude if c not in CLASS_TO_IDX]
        if unknown:
            raise ValueError(f"Unknown class name(s) to exclude: {unknown}. "
                             f"Valid classes: {CLASSES}")
        excluded_idx = {CLASS_TO_IDX[c] for c in exclude}
        kept_idx = [i for i in range(NUM_CLASSES) if i not in excluded_idx]
        remap = {old: new for new, old in enumerate(kept_idx)}  # old label -> new label
        entries = [(arr, remap[lbl]) for arr, lbl in entries if lbl not in excluded_idx]
        class_names = [CLASSES[i] for i in kept_idx]
        if not entries:
            raise RuntimeError("No videos left after applying --exclude.")
        print(f"Excluding {sorted(exclude)} -> training on {class_names}")
    else:
        class_names = list(CLASSES)

    random.seed(seed)
    random.shuffle(entries)

    n_val = max(1, int(len(entries) * val_split)) if len(entries) > 1 else 0
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    train_ds = DrowsinessDataset(train_entries, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_entries:
        val_ds = DrowsinessDataset(val_entries, stride=stride)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Videos: {len(train_entries)} train / {len(val_entries)} val")
    print(f"Windows: {len(train_ds)} train"
          + (f" / {len(val_ds)} val" if val_loader else ""))
    # --- previous version -------------------------------------------------
    # return train_loader, val_loader
    # ----------------------------------------------------------------------
    return train_loader, val_loader, class_names


if __name__ == "__main__":
    # Quick self-test
    # --- previous version: tr, va = make_dataloaders(batch_size=8) ---
    tr, va, names = make_dataloaders(batch_size=8)
    xb, yb = next(iter(tr))
    print("batch x:", xb.shape, "batch y:", yb.shape, "labels:", yb.tolist())
