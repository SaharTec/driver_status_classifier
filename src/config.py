"""
Shared configuration for the Deep-DMS project.
All stages (preprocessing, dataset, model, training) import from here so that
class order, paths and hyper-parameters stay consistent across the pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LABELS_FILE = PROCESSED_DIR / "labels.json"

# ---------------------------------------------------------------------------
# Classes - the index in this list IS the label used for training.
# Do not reorder once you have trained a model.
# ---------------------------------------------------------------------------
CLASSES = ["Alert", "Drowsy", "Sleeping", "Singing", "Yawning"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Alert logic - how the 6 raw classes map onto driver safety states.
#   ALERT  = driver is awake / engaged  -> never raises an alarm
#   DANGER = drowsiness signs           -> can raise an alarm (see src/alerts.py)
#   NEUTRAL = not drowsy, but not the focus of this system
# These lists are the single source of truth; change them here, not in code.
# ---------------------------------------------------------------------------
ALERT_CLASSES = ["Alert", "Singing"]              # talking / singing / awake
DANGER_CLASSES = ["Drowsy", "Sleeping", "Yawning"]  # the states that trigger sound

# Folder that holds the alarm recordings (.wav). Missing files fall back to a beep.
ALERTS_DIR = ROOT / "alerts"

# MediaPipe Tasks model (auto-downloaded on first run)
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Sequence / feature settings
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 60   # frames per sample (~2 seconds at 30 fps)
NUM_FEATURES = 7       # [EAR, MAR, yaw, pitch, roll, Δ_EAR, Δ_MAR]

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices (same ones used in the original main.py)
# ---------------------------------------------------------------------------
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 82, 312, 308, 317, 87]

# Head-pose: 6 landmarks (nose tip, chin, eye corners, mouth corners)
# matched against a generic 3D face model. solvePnP gives yaw/pitch/roll.
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
# Generic 3D face model points (in millimetres). Standard values used in
# countless head-pose tutorials; the actual scale does not matter because
# we only use the rotation, not the translation.
import numpy as _np
HEAD_POSE_3D_MODEL = _np.array([
    (0.0,    0.0,    0.0),     # nose tip
    (0.0,   -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0),   # left eye outer corner
    (225.0,  170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0,  -150.0, -125.0),  # right mouth corner
], dtype=_np.float64)
