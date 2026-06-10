# Driver Status Classifier (Deep-DMS)

A Driver Monitoring System (DMS) that classifies a driver's state from video and
raises an audible alarm when drowsiness is detected.

The pipeline uses **MediaPipe FaceMesh** to extract facial features from each
frame (Eye Aspect Ratio, Mouth Aspect Ratio and head pose), then feeds short
frame sequences into an **LSTM** classifier trained with PyTorch.

## Classes

The model predicts one of 6 driver states:

| Class        | Meaning                                   | Safety mapping |
|--------------|-------------------------------------------|----------------|
| `Alert`      | Awake, engaged, looking forward           | Safe           |
| `Singing`    | Talking / singing along, mouth moving     | Safe           |
| `Distracted` | Looking away (phone, side, down)          | Neutral        |
| `Drowsy`     | Heavy eyelids, slow blinks, head drooping | **Danger**     |
| `Sleeping`   | Eyes fully closed                         | **Danger**     |
| `Yawning`    | Yawning                                   | **Danger**     |

The `Danger` states trigger an audio alert (see [src/alerts.py](src/alerts.py)).
The class-to-safety mapping is defined in one place in [src/config.py](src/config.py).

## Features extracted per frame

`[EAR, MAR, yaw, pitch, roll]` — eye openness, mouth openness, and head
orientation. Sequences of `SEQUENCE_LENGTH` (30) frames (~1 second at 30 fps)
form a single training sample.

## Project structure

```
driver_status_classifier/
├── src/
│   ├── config.py        # classes, paths, hyper-parameters (single source of truth)
│   ├── preprocess.py    # video -> per-frame features (EAR/MAR/head pose)
│   ├── features.py      # feature extraction helpers
│   ├── dataset.py       # PyTorch Dataset over processed sequences
│   ├── model.py         # LSTM classifier
│   ├── alerts.py        # audio alarm logic
│   └── youtube_clips.py # helper for sourcing clips
├── train.py             # training entry point
├── main.py              # quick EAR/MAR plot for a single video
├── live_demo.py         # real-time webcam demo
├── trim_yawns.py        # utility for trimming yawning clips
├── models/              # trained weights (best_model.pth) + training curves
├── alerts/              # alarm .wav files (falls back to a beep if missing)
└── data/                # raw videos + processed features (git-ignored, local only)
```

> **Note:** `data/` and all video files (`*.mp4`, `*.avi`, ...) are intentionally
> **not** committed to git. Record/place your own videos locally — see below.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Add training videos

Place your recordings under `data/raw_videos/<ClassName>/` using the exact class
folder names (capital first letter):

```
data/raw_videos/
├── Alert/
├── Drowsy/
├── Sleeping/
├── Singing/
├── Distracted/
└── Yawning/
```

See [RECORDING_AND_TRAINING_GUIDE.txt](RECORDING_AND_TRAINING_GUIDE.txt) for
recommended recording setup (8–10 clips per class, 15–20s each, dashcam angle).

### 2. Preprocess videos into features

```bash
python src/preprocess.py
```

Runs face detection on every frame and saves features to `data/processed/`.

### 3. Train the model

```bash
python train.py --epochs 30 --batch-size 32 --lr 0.001
```

Saves the best model to `models/best_model.pth` and a learning-curve plot to
`models/training_curves.png`. Runs on CPU if no GPU is available.

### 4. Run the live demo

```bash
python live_demo.py
```

Real-time classification from the webcam, with audio alerts on danger states.

## License

Academic project — Software Engineering degree.
