"""
Record a short clip from the webcam and save it to data/raw_videos/<CLASS>/.
Usage:
    py record_clip.py --class Alert
    py record_clip.py --class Drowsy
Press q or ESC to stop recording.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import RAW_VIDEOS_DIR, CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls", required=True,
                        choices=CLASSES, help="class folder to save into")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = RAW_VIDEOS_DIR / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{args.cls}_{int(time.time())}.mp4"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"Could not open camera {args.camera}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(filename),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    print(f"Recording -> {filename}")
    print("Press q or ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        cv2.putText(frame, f"Recording: {args.cls}  (q to stop)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
