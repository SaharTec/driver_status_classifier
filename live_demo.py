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
                    SEQUENCE_LENGTH, NUM_FEATURES, CLASSES, MODELS_DIR)
from features import compute_ear_mar, compute_head_pose
from model import DrowsinessLSTM
from alerts import AlertSystem


def load_model(device):
    """Load best_model.pth into a DrowsinessLSTM. Returns (model, hidden_size).

    We don't store the hidden size with the weights, so we try a few common
    values until one loads. If none works, raise.
    """
    weights_path = MODELS_DIR / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained model at {weights_path}. Train one first with "
            "'python train.py ...'"
        )

    state = torch.load(weights_path, map_location=device)
    # infer hidden size from the LSTM weight matrix shape:
    # lstm.weight_ih_l0 has shape (4 * hidden_size, input_size)
    ih = state.get("lstm.weight_ih_l0")
    if ih is None:
        raise RuntimeError("Unexpected checkpoint format - missing lstm weights.")
    hidden_size = ih.shape[0] // 4
    num_classes = state["fc.weight"].shape[0]  # handles models trained with --exclude

    model = DrowsinessLSTM(hidden_size=hidden_size, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {weights_path}  "
          f"(hidden_size={hidden_size}, num_classes={num_classes})")
    return model


def main(camera_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)

    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {camera_index}")

    show_mesh = True
    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_valid = [0.0] * NUM_FEATURES
    alert_system = AlertSystem()  # turns predictions into spoken alarms

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
        results = face_mesh.process(rgb)

        ear_text, mar_text = "EAR: --", "MAR: --"
        pose_text = "yaw/pitch/roll: --"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            if show_mesh:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_tesselation_style(),
                )

            for idx in RIGHT_EYE_INDICES + LEFT_EYE_INDICES + MOUTH_INDICES:
                lm = face_landmarks.landmark[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2,
                           (0, 255, 0), -1)

            ear, mar = compute_ear_mar(face_landmarks, w, h)
            yaw, pitch, roll = compute_head_pose(face_landmarks, w, h)
            last_valid = [ear, mar, yaw, pitch, roll]
            ear_text = f"EAR: {ear:.3f}"
            mar_text = f"MAR: {mar:.3f}"
            pose_text = f"yaw {yaw:+.2f}  pitch {pitch:+.2f}  roll {roll:+.2f}"
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # always feed a frame so the buffer stays in real time (forward-fill
        # mirrors what preprocessing does on missed detections)
        feature_buffer.append(list(last_valid))

        # Predict once the buffer is full.
        if len(feature_buffer) == SEQUENCE_LENGTH:
            x = torch.tensor([list(feature_buffer)],
                             dtype=torch.float32, device=device)  # (1, 30, 5)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            top = int(probs.argmax())
            label = CLASSES[top]
            conf = float(probs[top])

            # run the alarm rules on this prediction (plays sound if needed)
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
            for i, name in enumerate(CLASSES):
                p = float(probs[i])
                y = y0 + i * (bar_h + 4)
                cv2.rectangle(frame, (x0, y), (x0 + bar_max_w, y + bar_h),
                              (50, 50, 50), -1)
                cv2.rectangle(frame, (x0, y),
                              (x0 + int(bar_max_w * p), y + bar_h),
                              (0, 200, 0) if i == top else (180, 180, 180),
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
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
