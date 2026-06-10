"""
Interactive segment labeller.

Plays every video in a source folder and lets you mark segments and assign
each one its OWN class label (Alert / Drowsy / Sleeping / Singing /
Distracted / Yawning). Each marked segment is trimmed with ffmpeg and written
straight into data/raw_videos/<ClassName>/, so a single video can produce
clips for several different classes. The original is moved to
data/raw_videos/_originals/<source>/ so it is preserved but ignored by
preprocess.py (the underscore-prefixed parent folder is not in CLASSES).

By default it reads from  data/data to sort/ . You can point it at any folder:
    python trim_yawns.py "data/raw_videos/Yawning"

After running, redo:
    python src/preprocess.py
    python train.py --epochs 25 --hidden-size 32 --batch-size 32 --lr 0.001

Controls:
    SPACE     play / pause
    a / d     step 1 frame back / forward
    j / l     seek -1 sec / +1 sec
    i         mark IN point at current frame
    1..6      mark OUT at current frame AND save the segment with that class
    u         cancel current IN mark
    z         undo the last saved segment
    n         move on to next video (saves anything you marked)
    q         quit immediately
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import cv2

# make the modules in src/ importable so we share the one list of classes
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_videos"
DEFAULT_SOURCE = ROOT / "data" / "data to sort" / "Dash"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# digit '1'.. -> class name. Only the first 9 classes are reachable by keypad.
DIGIT_TO_CLASS = {str(i + 1): name for i, name in enumerate(CLASSES[:9])}
LEGEND = "  ".join(f"[{d}]{name}" for d, name in DIGIT_TO_CLASS.items())


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found on PATH.")
        print("  Install:  winget install Gyan.FFmpeg")
        sys.exit(1)


def trim_segment(src, start_sec, end_sec, dst):
    """Cut src from start_sec to end_sec and write to dst (frame-accurate)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-an",
        str(dst),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame


def play_and_mark(video_path):
    """
    Returns (segments, quit_all) where segments is a list of
    (start_s, end_s, label_name).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  could not open {video_path}")
        return [], False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"  no frames in {video_path}")
        return [], False

    paused = True
    in_frame = None
    segments = []
    current = 0
    quit_all = False
    need_seek = True  # whether to set POS_FRAMES on next iteration

    while True:
        if need_seek or paused:
            ok, frame = _read_frame(cap, current)
            need_seek = False
        else:
            ok, frame = cap.read()
            if not ok:
                current = max(0, total - 1)
                paused = True
                need_seek = True
                continue
            current = min(current + 1, total - 1)

        if not ok:
            break

        time_now = current / fps
        time_end = total / fps
        h = frame.shape[0]

        cv2.putText(frame,
                    f"{video_path.name}  {time_now:5.2f}s / {time_end:5.2f}s   "
                    f"frame {current}/{total - 1}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        if in_frame is not None:
            cv2.putText(frame,
                        f"IN at {in_frame / fps:5.2f}s  "
                        f"(press 1-{len(DIGIT_TO_CLASS)} to label OUT)",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        if segments:
            last = segments[-1]
            cv2.putText(frame,
                        f"saved {len(segments)} segment(s)  "
                        f"last: {last[2]} "
                        f"{last[0]:.2f}-{last[1]:.2f}s",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 200, 0), 2)
        cv2.putText(frame, LEGEND, (10, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame,
                    "[space]play [a/d]frame [j/l]+-1s  "
                    "[i]IN  [1-6]label OUT  [u]undo IN  [z]undo seg  "
                    "[n]next  [q]quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

        cv2.imshow("Segment labeller", frame)
        delay = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key == 255:  # no key
            if paused:
                continue

        ch = chr(key) if key != 255 else ""

        if key == ord('q'):
            quit_all = True
            break
        if key == ord('n'):
            break
        if key == ord(' '):
            paused = not paused
            need_seek = paused
            continue
        if key == ord('a'):
            paused = True
            current = max(0, current - 1)
            need_seek = True
            continue
        if key == ord('d'):
            paused = True
            current = min(total - 1, current + 1)
            need_seek = True
            continue
        if key == ord('j'):
            paused = True
            current = max(0, current - int(fps))
            need_seek = True
            continue
        if key == ord('l'):
            paused = True
            current = min(total - 1, current + int(fps))
            need_seek = True
            continue
        if key == ord('i'):
            in_frame = current
            paused = True
            need_seek = True
            print(f"  IN  @ {current / fps:.2f}s")
            continue
        if ch in DIGIT_TO_CLASS:
            label = DIGIT_TO_CLASS[ch]
            if in_frame is None:
                print("  press 'i' first to mark IN")
                paused = True
                need_seek = True
                continue
            if current <= in_frame:
                print("  OUT must be after IN")
                paused = True
                need_seek = True
                continue
            segments.append((in_frame / fps, current / fps, label))
            print(f"  OUT @ {current / fps:.2f}s   -> {label} "
                  f"{in_frame / fps:.2f}s -> {current / fps:.2f}s")
            in_frame = None
            paused = True
            need_seek = True
            continue
        if key == ord('u'):
            in_frame = None
            paused = True
            need_seek = True
            print("  cancelled IN mark")
            continue
        if key == ord('z'):
            if segments:
                dropped = segments.pop()
                print(f"  removed last segment: {dropped[2]} "
                      f"{dropped[0]:.2f}-{dropped[1]:.2f}s")
            else:
                print("  no segments to undo")
            paused = True
            need_seek = True
            continue

    cap.release()
    return segments, quit_all


def main():
    check_ffmpeg()

    # source folder: CLI arg or the default "data/data to sort"
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    if not source.exists():
        print(f"ERROR: source folder not found: {source}")
        print("  Pass one explicitly:  python trim_yawns.py \"data/my videos\"")
        sys.exit(1)

    orig_dir = RAW / "_originals" / source.name
    orig_dir.mkdir(parents=True, exist_ok=True)

    # recurse, so nested layouts like Dash/Female, Dash/Male are picked up
    videos = sorted(p for p in source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                    and "_originals" not in p.parts)
    if not videos:
        print(f"No videos in {source}")
        return

    print(f"Source: {source}")
    print(f"Found {len(videos)} videos to label.\n")
    print("Controls:")
    print("  SPACE    play / pause")
    print("  a / d    step 1 frame back / forward")
    print("  j / l    seek -1s / +1s")
    print("  i        mark IN at current frame")
    print(f"  1..{len(DIGIT_TO_CLASS)}    mark OUT + label segment "
          f"({LEGEND.strip()})")
    print("  u        undo current IN")
    print("  z        undo last saved segment")
    print("  n        next video")
    print("  q        quit\n")

    for idx, video in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] {video.name}")
        segments, quit_all = play_and_mark(video)

        if segments:
            # per-class counter so filenames within this video stay unique
            counts = {}
            for (a, b, label) in segments:
                counts[label] = counts.get(label, 0) + 1
                dst_dir = RAW / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{video.stem}_{label}{counts[label]}.mp4"
                try:
                    trim_segment(video, a, b, dst)
                    print(f"  -> {label}/{dst.name}  ({b - a:.2f}s)")
                except subprocess.CalledProcessError as exc:
                    print(f"  ffmpeg failed for {label} segment: {exc}")
            # keep the sub-folder layout (Female/, Male/) under _originals
            rel = video.relative_to(source)
            dest = orig_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(video), str(dest))
            print(f"  original moved to {dest}\n")
        else:
            print("  no segments marked; leaving original in place.\n")

        if quit_all:
            print("Quit pressed - stopping.")
            break

    cv2.destroyAllWindows()
    print("\nDone. Next steps:")
    print("  python src/preprocess.py")
    print("  python train.py --epochs 25 --hidden-size 32 "
          "--batch-size 32 --lr 0.001")


if __name__ == "__main__":
    main()
