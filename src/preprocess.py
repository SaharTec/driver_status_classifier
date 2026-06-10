"""
Stage A - Data Engineering.

Walks over data/raw_videos/<ClassName>/*.mp4, extracts [EAR, MAR] features
for every frame, and saves them as compact .npy arrays in data/processed/.
A labels.json file maps every saved .npy to its multi-class label index.

Expected input layout (one sub-folder per class):

    data/raw_videos/
        Alert/      alert_01.mp4, alert_02.mp4, ...
        Drowsy/     ...
        Sleeping/   ...
        Singing/    ...
        Distracted/ ...
        Yawning/    ...

Run from the project root:   python src/preprocess.py
"""
import json
import sys

# Windows consoles default to cp1252, which can't print the Hebrew project
# path or unicode arrows. Force UTF-8 so progress prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np

from config import (RAW_VIDEOS_DIR, PROCESSED_DIR, LABELS_FILE,
                    CLASSES, CLASS_TO_IDX, SEQUENCE_LENGTH)
from features import extract_features_from_video

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_VIDEOS_DIR.exists():
        print(f"[!] Raw videos folder not found: {RAW_VIDEOS_DIR}")
        return

    labels = {}
    total_videos = 0

    for class_name in CLASSES:
        class_dir = RAW_VIDEOS_DIR / class_name
        if not class_dir.exists():
            print(f"[i] No folder for class '{class_name}' (skipping).")
            continue

        videos = [p for p in sorted(class_dir.iterdir())
                  if p.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            print(f"[i] No videos in '{class_name}'.")
            continue

        print(f"\n=== Class '{class_name}' ({len(videos)} videos) ===")
        for video_path in videos:
            print(f"  -> {video_path.name}")
            features = extract_features_from_video(video_path, verbose=True)

            if features.shape[0] < SEQUENCE_LENGTH:
                print(f"     [!] only {features.shape[0]} frames "
                      f"(< {SEQUENCE_LENGTH}); skipping.")
                continue

            out_name = f"{class_name}__{video_path.stem}.npy"
            out_path = PROCESSED_DIR / out_name
            np.save(out_path, features)

            labels[out_name] = CLASS_TO_IDX[class_name]
            total_videos += 1
            print(f"     saved {out_name}  shape={features.shape}")

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {total_videos} videos processed -> {PROCESSED_DIR}")
    print(f"Labels written to {LABELS_FILE}")


if __name__ == "__main__":
    main()
