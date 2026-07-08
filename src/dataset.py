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

    def __init__(self, entries, seq_len=SEQUENCE_LENGTH, stride=1, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        self.arrays = [arr for arr, _ in entries]
        self.video_labels = [lbl for _, lbl in entries]

        # Sliding window: each (video_idx, start_frame) pair is one training sample.
        # stride > 1 skips frames between windows, producing fewer but more
        # independent samples (useful when training data is large).
        self.samples = []
        for v_idx, arr in enumerate(self.arrays):
            last_start = arr.shape[0] - seq_len
            for start in range(0, last_start + 1, stride):
                self.samples.append((v_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_idx, start = self.samples[idx]
        window = self.arrays[v_idx][start:start + self.seq_len].copy()  # (seq_len, 5)

        # Data augmentation: add tiny Gaussian noise during training to make
        # the model less sensitive to small variations in feature values.
        if self.augment:
            window += np.random.normal(0, 0.01, window.shape).astype(np.float32)

        # Append frame-to-frame deltas for EAR and MAR (columns 0 and 1).
        # These capture the rate of change (e.g. how quickly eyes are closing)
        # which the raw values alone don't convey.
        delta = np.zeros((self.seq_len, 2), dtype=np.float32)
        delta[1:] = window[1:, :2] - window[:-1, :2]
        window = np.concatenate([window, delta], axis=1)   # (seq_len, 7)

        x = torch.from_numpy(window)
        y = torch.tensor(self.video_labels[v_idx], dtype=torch.long)
        return x, y


def make_dataloaders(batch_size=32, val_split=0.2, stride=1, seed=42,
                     exclude=None, merge=None):
    """
    Split videos into train/val, build sliding-window datasets and return
    (train_loader, val_loader, class_names).

    `exclude`: list of class names to drop entirely (e.g. ["Sleeping"]).
    `merge`:   (src, dst) tuple OR list of (src, dst) tuples.
               Each src is relabelled as dst and disappears as a separate class.
    merge and exclude can be combined (exclude is applied on original labels).
    Labels are always remapped to a contiguous 0..K-1 range.
    """
    entries = _load_entries()
    if not entries:
        raise RuntimeError("No usable videos found in data/processed.")

    if merge or exclude:
        # Build merge map on original label indices
        raw_map = list(range(NUM_CLASSES))
        if merge:
            pairs = [merge] if isinstance(merge[0], str) else list(merge)
            for src_name, dst_name in pairs:
                for name in (src_name, dst_name):
                    if name not in CLASS_TO_IDX:
                        raise ValueError(f"Unknown class '{name}'. Valid: {CLASSES}")
                raw_map[CLASS_TO_IDX[src_name]] = CLASS_TO_IDX[dst_name]

        # Filter excluded classes (by original label), then apply merge map
        excluded_idx = {CLASS_TO_IDX[c] for c in (exclude or [])}
        unknown = [c for c in (exclude or []) if c not in CLASS_TO_IDX]
        if unknown:
            raise ValueError(f"Unknown class(es) to exclude: {unknown}. Valid: {CLASSES}")
        entries = [(arr, raw_map[lbl]) for arr, lbl in entries if lbl not in excluded_idx]
        if not entries:
            raise RuntimeError("No videos left after applying filters.")

        # Remap to contiguous 0..K-1
        remaining = sorted(set(lbl for _, lbl in entries))
        compact = {old: new for new, old in enumerate(remaining)}
        entries = [(arr, compact[lbl]) for arr, lbl in entries]
        class_names = [CLASSES[i] for i in remaining]
        if merge:
            print(f"Merging {pairs} -> training on {class_names}")
        if exclude:
            print(f"Excluding {sorted(exclude)} -> training on {class_names}")
    else:
        class_names = list(CLASSES)

    random.seed(seed)
    random.shuffle(entries)

    n_val = max(1, int(len(entries) * val_split)) if len(entries) > 1 else 0
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    train_ds = DrowsinessDataset(train_entries, stride=stride, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_entries:
        val_ds = DrowsinessDataset(val_entries, stride=stride)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Videos: {len(train_entries)} train / {len(val_entries)} val")
    print(f"Windows: {len(train_ds)} train"
          + (f" / {len(val_ds)} val" if val_loader else ""))
    return train_loader, val_loader, class_names


if __name__ == "__main__":
    tr, va, names = make_dataloaders(batch_size=8)
    xb, yb = next(iter(tr))
    print("batch x:", xb.shape, "batch y:", yb.shape, "labels:", yb.tolist())
