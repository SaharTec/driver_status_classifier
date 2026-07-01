"""
Download a YouTube video and slice it into fixed-length training clips for a
single driver-state class.

Each clip is written to data/raw_videos/<Class>/ and named <Class>NNN.mp4,
continuing the numbering of whatever is already in that folder (nothing is
overwritten). The class is decided purely by the target folder, so every clip
produced by one run gets the same label - point the tool at a video (or the
time ranges within it) that show ONE behavior.

Requirements: yt-dlp and ffmpeg must be on PATH.

Examples
--------
# Whole video -> Dazzled clips, default 18s each
python src/youtube_clips.py "https://youtu.be/VIDEO" --label Dazzled

# Only the parts where the behavior actually happens
python src/youtube_clips.py "https://youtu.be/VIDEO" --label Drowsy \
        --range 0:30-1:10 --range 2:05-2:50

# Different clip length (Yawning guide wants 20-25s)
python src/youtube_clips.py "https://youtu.be/VIDEO" --label Yawning --clip-len 22
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, RAW_VIDEOS_DIR  # noqa: E402


def parse_timestamp(text: str) -> float:
    """Accept seconds ('95'), 'MM:SS' or 'HH:MM:SS' and return seconds."""
    text = text.strip()
    if not text:
        raise ValueError("empty timestamp")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad timestamp: {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_range(text: str) -> tuple[float, float]:
    """Parse 'START-END' (e.g. '0:30-1:10') into (start_sec, end_sec)."""
    # split on the dash that separates the two timestamps (not the ':' inside)
    m = re.match(r"^\s*(.+?)\s*-\s*(.+?)\s*$", text)
    if not m:
        raise ValueError(f"range must look like START-END, got {text!r}")
    start, end = parse_timestamp(m.group(1)), parse_timestamp(m.group(2))
    if end <= start:
        raise ValueError(f"end must be after start in {text!r}")
    return start, end


def next_index(folder: Path, label: str) -> int:
    """Return the next free NNN so existing <label>NNN.mp4 files are preserved."""
    pattern = re.compile(rf"^{re.escape(label)}(\d+)\.mp4$", re.IGNORECASE)
    highest = 0
    for f in folder.glob("*.mp4"):
        m = pattern.match(f.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def download(url: str, dest_dir: Path) -> Path:
    """Download the best mp4 (<=720p keeps faces sharp but files small)."""
    out_template = str(dest_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    print(f"Downloading {url} ...")
    subprocess.run(cmd, check=True)
    files = list(dest_dir.glob("source.*"))
    if not files:
        raise RuntimeError("yt-dlp finished but produced no file")
    return files[0]


def video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def cut_clip(source: Path, start: float, length: float, out_path: Path) -> None:
    """Re-encode a clip so cut points are frame-accurate (matters for short clips)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{length:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-an",                       # drop audio - training only uses video frames
        "-loglevel", "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def slice_segment(source: Path, seg_start: float, seg_end: float,
                  clip_len: float, min_len: float,
                  out_dir: Path, label: str, index: int) -> int:
    """Cut [seg_start, seg_end) into clip_len pieces. Returns the next index."""
    t = seg_start
    while t < seg_end:
        length = min(clip_len, seg_end - t)
        if length < min_len:
            print(f"  skipping trailing {length:.1f}s piece (< {min_len:.0f}s min)")
            break
        out_path = out_dir / f"{label}{index:03d}.mp4"
        cut_clip(source, t, length, out_path)
        print(f"  wrote {out_path.name}  ({t:.1f}s -> {t + length:.1f}s)")
        index += 1
        t += clip_len
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("--label", required=True,
                    help=f"target class folder. Known classes: {', '.join(CLASSES)}")
    ap.add_argument("--range", action="append", dest="ranges", metavar="START-END",
                    help="time range to slice (repeatable). Omit to use whole video.")
    ap.add_argument("--clip-len", type=float, default=18.0,
                    help="seconds per clip (default 18)")
    ap.add_argument("--min-len", type=float, default=2.0,
                    help="discard a trailing piece shorter than this (default 2s; "
                         "preprocess needs >= ~1s of detectable face)")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the full downloaded video next to the clips")
    args = ap.parse_args()

    if args.label not in CLASSES:
        print(f"WARNING: '{args.label}' is not in config.CLASSES "
              f"({', '.join(CLASSES)}). Continuing, but training ignores "
              f"folders whose name is not a known class.")

    out_dir = RAW_VIDEOS_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        source = download(args.url, Path(tmp))
        duration = video_duration(source)
        print(f"Downloaded {duration:.1f}s video -> slicing into {args.clip_len:.0f}s clips")

        if args.ranges:
            segments = [parse_range(r) for r in args.ranges]
        else:
            segments = [(0.0, duration)]

        index = next_index(out_dir, args.label)
        start_index = index
        for seg_start, seg_end in segments:
            seg_end = min(seg_end, duration)
            print(f"Segment {seg_start:.1f}s -> {seg_end:.1f}s")
            index = slice_segment(source, seg_start, seg_end, args.clip_len,
                                  args.min_len, out_dir, args.label, index)

        if args.keep_source:
            kept = out_dir / f"_source_{source.name}"
            kept.write_bytes(source.read_bytes())
            print(f"Kept source video -> {kept}")

    made = index - start_index
    print(f"\nDone: {made} clip(s) written to {out_dir}")
    if made:
        print("Next: python src/preprocess.py   then   python train.py")
    else:
        print("No clips written - check your time ranges.")


if __name__ == "__main__":
    main()
