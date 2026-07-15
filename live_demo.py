"""
Live camera demo: open the webcam, run MediaPipe FaceMesh, extract the same
[EAR, MAR, yaw, pitch, roll] features used during training, and feed a rolling
window of SEQUENCE_LENGTH frames into the trained DrowsinessLSTM to predict
the current driver state in real time.

Controls:
    q / ESC  - quit
    m        - toggle full face mesh on/off (keeps the EAR/MAR points)

Run:
    python live_demo.py
"""
import argparse
import os
import sys
from collections import deque

# Windows consoles default to cp1252 and can't print the Hebrew path
# this project lives under. Force UTF-8 so startup prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

# make the modules in src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import (RIGHT_EYE_INDICES, LEFT_EYE_INDICES, MOUTH_INDICES,
                    SEQUENCE_LENGTH, CLASSES, MODELS_DIR)
from features import compute_ear_mar, compute_head_pose, make_face_detector
from model import DrowsinessLSTM
from alerts import AlertSystem



def load_model(device):
    """Load best_model.pth. Returns (model, class_names).

    Handles two checkpoint formats:
    - New (dict):  {"model_type": "lstm", "state_dict": ..., "class_names": [...], ...}
    - Legacy (bare state_dict): old LSTM checkpoints saved before the dict format.
    """
    weights_path = MODELS_DIR / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained model at {weights_path}. Train one first with "
            "'python train.py ...'"
        )

    raw = torch.load(weights_path, map_location=device)

    if isinstance(raw, dict) and "model_type" in raw:
        model_type = raw["model_type"]
        state = raw["state_dict"]
        class_names = raw.get("class_names", list(CLASSES))
        num_classes = len(class_names)
        num_features = raw.get("num_features", 7)  # default 7 for older checkpoints

        hidden_size = raw.get("hidden_size", 128)
        num_layers = raw.get("num_layers", 2)
        model = DrowsinessLSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes).to(device)
    else:
        # Legacy bare state_dict — infer LSTM config from weight shapes
        state = raw
        ih = state.get("lstm.weight_ih_l0")
        if ih is None:
            raise RuntimeError("Unexpected checkpoint format — missing lstm weights.")
        hidden_size = ih.shape[0] // 4
        num_features = ih.shape[1]   # infer from weight matrix, not from config
        num_classes = state["fc.weight"].shape[0]
        num_layers = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))
        model = DrowsinessLSTM(input_size=num_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes).to(device)
        model_type = "lstm"
        class_names = list(CLASSES)

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_type.upper()} from {weights_path}  "
          f"(num_classes={num_classes}, classes={class_names})")
    return model, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, class_names = load_model(device)

    detector = make_face_detector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {args.camera}")

    show_mesh = True

    # Rolling feature buffer — holds the last SEQUENCE_LENGTH frames.
    # Once full, its contents are fed to the model as one prediction window.
    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Probability smoothing — average the last 45 raw prediction vectors
    # so single-frame noise doesn't flip the displayed label.
    pred_buffer = deque(maxlen=45)

    # EAR smoothing — average the last 5 EAR values to absorb normal blinks,
    # which would otherwise briefly spike the feature and confuse the model.
    ear_history = deque(maxlen=5)

    # Forward-fill: when no face is detected we reuse the previous frame's
    # features so the buffer keeps advancing in real time.
    last_valid = [0.0] * 5

    alert_system = AlertSystem()  # plays spoken alarms when drowsiness persists

    # Hysteresis state — prevents rapid flipping between labels.
    # A new label must win SWITCH_FRAMES consecutive frames before we commit to it.
    stable_label = None
    stable_conf = 0.0
    candidate_label = None   # label currently being "auditioned"
    candidate_frames = 0     # how many frames in a row the candidate has won
    SWITCH_FRAMES = 30
    DROWSY_MIN_PROB = 0.50   # Drowsy must reach 50% averaged confidence to trigger

    print("Camera opened. Press 'q' or ESC to quit, 'm' to toggle the mesh.")
    print(f"Need {SEQUENCE_LENGTH} frames before predictions start...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        ear_text, mar_text = "EAR: --", "MAR: --"
        pose_text = "yaw/pitch/roll: --"

        # --- Feature extraction ------------------------------------------
        # MediaPipe gives us 478 face landmarks; we compute three things:
        #   EAR  — Eye Aspect Ratio: drops when eyes close (drowsiness signal)
        #   MAR  — Mouth Aspect Ratio: rises when mouth opens (yawn signal)
        #   yaw/pitch/roll — head orientation (catches head nodding / tilting)
        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            if show_mesh:
                for lm in face_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                               1, (80, 80, 80), -1)

            for idx in RIGHT_EYE_INDICES + LEFT_EYE_INDICES + MOUTH_INDICES:
                lm = face_landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2,
                           (0, 255, 0), -1)

            ear, mar = compute_ear_mar(face_landmarks, w, h)
            yaw, pitch, roll = compute_head_pose(face_landmarks, w, h)
            ear_history.append(ear)
            smoothed_ear = float(np.mean(ear_history))
            last_valid = [smoothed_ear, mar, yaw, pitch, roll]
            ear_text = f"EAR: {smoothed_ear:.3f}"
            mar_text = f"MAR: {mar:.3f}"
            pose_text = f"yaw {yaw:+.2f}  pitch {pitch:+.2f}  roll {roll:+.2f}"
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Always push a frame (forward-fill when face is missing) so the
        # buffer tracks real time and predictions stay in sync with the video.
        feature_buffer.append(list(last_valid))

        # --- Model inference ---------------------------------------------
        # Wait until we have a full window of SEQUENCE_LENGTH frames, then
        # build the 7-feature input (5 raw + 2 frame-to-frame deltas) and
        # run the LSTM.
        if len(feature_buffer) == SEQUENCE_LENGTH:
            buf = np.array(list(feature_buffer))               # (60, 5)
            delta = np.zeros((len(buf), 2), dtype=np.float32)
            delta[1:] = buf[1:, :2] - buf[:-1, :2]            # Δ_EAR, Δ_MAR
            buf_full = np.concatenate([buf, delta], axis=1)    # (60, 7)
            x = torch.tensor([buf_full], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            # --- Probability smoothing -----------------------------------
            # Average the last 45 prediction vectors before deciding the label.
            pred_buffer.append(probs)
            avg_probs = np.mean(list(pred_buffer), axis=0)

            # --- Drowsy confidence threshold -----------------------------
            # Only classify as Drowsy when its averaged probability clears
            # DROWSY_MIN_PROB; otherwise treat the frame as Alert to reduce
            # false alarms caused by the model's residual Drowsy bias.
            raw_idx = int(avg_probs.argmax())
            if (class_names[raw_idx] == "Drowsy"
                    and avg_probs[raw_idx] < DROWSY_MIN_PROB
                    and "Alert" in class_names):
                top_idx = class_names.index("Alert")
            else:
                top_idx = raw_idx
            top_label = class_names[top_idx]
            top_conf = float(avg_probs[top_idx])

            # --- Hysteresis ----------------------------------------------
            # The displayed label (stable_label) only changes once a new
            # label has been the top prediction for SWITCH_FRAMES frames in
            # a row.  Single-frame blips never reach the screen.
            if stable_label is None:
                stable_label = top_label
                stable_conf = top_conf
            elif top_label != stable_label:
                if top_label == candidate_label:
                    candidate_frames += 1
                else:
                    candidate_label = top_label
                    candidate_frames = 1
                if candidate_frames >= SWITCH_FRAMES:
                    stable_label = candidate_label
                    stable_conf = top_conf
                    candidate_label = None
                    candidate_frames = 0
            else:
                stable_conf = top_conf
                candidate_label = None
                candidate_frames = 0
            label = stable_label
            conf = stable_conf

            # --- Alert system --------------------------------------------
            # Passes the stable label to AlertSystem; it plays a sound when
            # drowsiness has persisted long enough to trigger an alarm.
            banner = alert_system.update(label)
            if banner:
                cv2.rectangle(frame, (0, 85), (w, 130), (0, 0, 255), -1)
                cv2.putText(frame, banner, (20, 117),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # big label at top
            cv2.rectangle(frame, (10, 10), (10 + 380, 10 + 70),
                          (0, 0, 0), -1)
            cv2.putText(frame, f"{label}  ({conf*100:.0f}%)",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255), 3)

            # mini bar chart of all classes (right side)
            bar_h = 18
            bar_max_w = 220
            x0 = w - bar_max_w - 20
            y0 = 20
            for i, name in enumerate(class_names):
                p = float(avg_probs[i])
                y = y0 + i * (bar_h + 4)
                cv2.rectangle(frame, (x0, y), (x0 + bar_max_w, y + bar_h),
                              (50, 50, 50), -1)
                cv2.rectangle(frame, (x0, y),
                              (x0 + int(bar_max_w * p), y + bar_h),
                              (0, 200, 0) if i == top_idx else (180, 180, 180),
                              -1)
                cv2.putText(frame, f"{name} {p*100:4.0f}%",
                            (x0 + 5, y + bar_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
        else:
            cv2.putText(frame,
                        f"Buffering... {len(feature_buffer)}/{SEQUENCE_LENGTH}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

        cv2.putText(frame, ear_text, (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, mar_text, (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, pose_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Driver state - live (q=quit, m=toggle mesh)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mesh = not show_mesh

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
