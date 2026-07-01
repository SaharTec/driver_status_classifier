"""
Feature extraction: turn a video into a (num_frames, 5) array of
[EAR, MAR, yaw, pitch, roll].

EAR   = Eye Aspect Ratio  -> drops when eyes close
MAR   = Mouth Aspect Ratio -> rises when the mouth opens (yawning / talking)
yaw   = head rotation left/right   (normalized to ~[-1, 1])
pitch = head rotation up/down      (normalized to ~[-1, 1])
roll  = head tilt clockwise/ccw    (normalized to ~[-1, 1])

Head pose lets the model distinguish classes that look identical in EAR/MAR
alone (e.g. head tilt during Drowsy vs Alert).

This logic is the same maths as the original main.py, refactored so both the
preprocessing script and a real-time demo can reuse it.
"""
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (RIGHT_EYE_INDICES, LEFT_EYE_INDICES, MOUTH_INDICES,
                    HEAD_POSE_LANDMARKS, HEAD_POSE_3D_MODEL,
                    FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL)


def _ensure_model():
    """Download face_landmarker.task on first use."""
    if not FACE_LANDMARKER_MODEL.exists():
        print(f"Downloading face_landmarker.task to {FACE_LANDMARKER_MODEL} ...")
        FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("Download complete.")


def make_face_detector():
    """Create and return a MediaPipe FaceLandmarker (Tasks API)."""
    _ensure_model()
    base_options = mp_python.BaseOptions(
        model_asset_path=str(FACE_LANDMARKER_MODEL))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _get_coords(landmarks, indices, img_w, img_h):
    return np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h]
         for i in indices]
    )


def _aspect_ratio(points):
    """Generic 6-point aspect ratio: (|p1-p5| + |p2-p4|) / (2*|p0-p3|)."""
    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h = np.linalg.norm(points[0] - points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear_mar(face_landmarks, img_w, img_h):
    """Return (ear, mar) for a single detected face."""
    right_eye = _get_coords(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)
    left_eye = _get_coords(face_landmarks, LEFT_EYE_INDICES, img_w, img_h)
    mouth = _get_coords(face_landmarks, MOUTH_INDICES, img_w, img_h)

    ear = (_aspect_ratio(right_eye) + _aspect_ratio(left_eye)) / 2.0
    mar = _aspect_ratio(mouth)
    return ear, mar


def compute_head_pose(face_landmarks, img_w, img_h):
    """
    Return (yaw, pitch, roll) for a single detected face, normalized to
    roughly [-1, 1] so they have a similar scale to EAR/MAR.

    Uses cv2.solvePnP against a generic 3D face model.
    """
    image_points = np.array(
        [[face_landmarks[i].x * img_w,
          face_landmarks[i].y * img_h]
         for i in HEAD_POSE_LANDMARKS],
        dtype=np.float64,
    )

    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _tvec = cv2.solvePnP(
        HEAD_POSE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy < 1e-6:
        # gimbal-lock fallback
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])

    # Normalize radians to roughly [-1, 1] (pi ~= 3.14).
    return yaw / np.pi, pitch / np.pi, roll / np.pi


def compute_all_features(face_landmarks, img_w, img_h):
    """Return [ear, mar, yaw, pitch, roll] for one frame's face."""
    ear, mar = compute_ear_mar(face_landmarks, img_w, img_h)
    yaw, pitch, roll = compute_head_pose(face_landmarks, img_w, img_h)
    return [ear, mar, yaw, pitch, roll]


def extract_features_from_video(video_path, verbose=False):
    """
    Process one video file and return a float32 array of shape (num_frames, 5).

    Frames where no face is detected reuse the previous valid value
    (forward-fill), or zeros if no face has been seen yet.
    """
    detector = make_face_detector()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        raise IOError(f"Could not open video: {video_path}")

    features = []
    last_valid = [0.0, 0.0, 0.0, 0.0, 0.0]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            last_valid = compute_all_features(
                result.face_landmarks[0], img_w, img_h)

        features.append(list(last_valid))
        frame_count += 1
        if verbose and frame_count % 50 == 0:
            print(f"    processed {frame_count} frames...")

    cap.release()
    detector.close()
    return np.asarray(features, dtype=np.float32)
