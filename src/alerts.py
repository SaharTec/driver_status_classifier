"""
Driver-state alert logic for the live demo.

This module turns the model's per-frame prediction into spoken alarms, using
the rules the project owner defined:

  * Sleeping  - asleep continuously for > 5 s            -> 1 recording (wake up)
  * Drowsy    - head nodding continuously for > 5 s      -> 2 recordings
                                                           (wake up, then pull over)
  * Yawning   - more than 3 yawns inside a 30 s window   -> 1 recording (take a rest)
  * After any alert has fired, while that danger state keeps happening we
    replay the matching recording (ignoring the time threshold), spaced out by
    a cooldown so it does not spam.
  * When the driver goes back to an ALERT state (Alert / Singing) for a short
    while, everything resets and the alarms can arm again from scratch.

All timing is wall-clock based (time.time()), so it does not depend on the
camera frame rate.

Sound playback runs on a background thread so the video loop never blocks. If a
.wav file is missing it falls back to a system beep, so the alarm is always
audible even before you have recorded the real messages.
"""
import queue
import threading
import time
from collections import deque
from pathlib import Path

from config import ALERT_CLASSES, DANGER_CLASSES, ALERTS_DIR

try:
    import winsound  # Windows only; this project runs on Windows
except ImportError:  # pragma: no cover - keeps import working off-Windows
    winsound = None


# ---------------------------------------------------------------------------
# Tuning constants - THESE ARE "THE RULES". Edit here to change behaviour.
# ---------------------------------------------------------------------------
SLEEP_TRIGGER_SECONDS = 5.0     # asleep this long (continuous) -> alarm
DROWSY_TRIGGER_SECONDS = 5.0    # nodding this long (continuous) -> alarm
YAWN_WINDOW_SECONDS = 30.0      # count yawns inside this sliding window
YAWN_TRIGGER_COUNT = 3          # fire when yawns are MORE THAN this (i.e. 4+)

STREAK_GRACE_SECONDS = 0.5      # ignore brief mis-detections inside a streak
REPEAT_COOLDOWN_SECONDS = 8.0   # min gap between repeats of the same alarm
RECOVERY_SECONDS = 1.5          # awake this long -> reset / re-arm everything
BANNER_SECONDS = 4.0            # how long the on-screen warning text stays up

# Recording file names expected inside ALERTS_DIR (see alerts/README.md).
SOUND_FILES = {
    "sleeping":        "sleeping_alert.wav",
    "yawning":         "yawning_alert.wav",
    "drowsy_wake":     "drowsy_wake.wav",
    "drowsy_pullover": "drowsy_pullover.wav",
}


class AlertPlayer:
    """Plays .wav files one after another on a background thread.

    Queueing keeps the caller non-blocking and lets us play several recordings
    in sequence (used by the Drowsy rule, which plays two).
    """

    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def play_sequence(self, paths):
        for p in paths:
            self._q.put(p)

    def _worker(self):
        while True:
            path = self._q.get()
            try:
                self._play_blocking(path)
            except Exception as exc:  # never let a bad file kill the thread
                print(f"[alert] could not play {path}: {exc}")

    @staticmethod
    def _play_blocking(path):
        path = Path(path)
        if winsound is None:
            print("\a", end="", flush=True)  # terminal bell fallback
            return
        if path.exists():
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        else:
            # Recording not supplied yet - beep so the alarm is still audible.
            winsound.Beep(1000, 600)


class _Streak:
    """Tracks how long a condition has been continuously true.

    A short gap (<= grace) does not break the streak, which absorbs the
    frame-to-frame jitter typical of the classifier.
    """

    def __init__(self, grace):
        self.grace = grace
        self.start = None
        self.last_active = None

    def update(self, active, now):
        """Returns True on the frame a brand-new streak begins."""
        started = False
        if active:
            if self.start is None:
                self.start = now
                started = True
            self.last_active = now
        elif self.last_active is not None and now - self.last_active > self.grace:
            self.start = None
        return started

    def duration(self, now):
        return 0.0 if self.start is None else now - self.start

    def reset(self):
        self.start = None
        self.last_active = None


class AlertSystem:
    """Feeds it the current label every frame; it raises alarms by the rules."""

    def __init__(self, player=None, alerts_dir=ALERTS_DIR):
        self.player = player or AlertPlayer()
        self.dir = Path(alerts_dir)

        self.sleep_streak = _Streak(STREAK_GRACE_SECONDS)
        self.drowsy_streak = _Streak(STREAK_GRACE_SECONDS)
        self.yawn_streak = _Streak(STREAK_GRACE_SECONDS)
        self.awake_streak = _Streak(0.0)
        self.yawn_events = deque()  # timestamps of yawn onsets in the window

        # "armed" = threshold already met once; keep repeating while it lasts.
        self.armed = {"sleeping": False, "drowsy": False, "yawning": False}
        self.last_fire = {"sleeping": 0.0, "drowsy": 0.0, "yawning": 0.0}

        self.banner = ""        # short warning text for the UI to draw
        self.banner_until = 0.0

    # -- helpers -----------------------------------------------------------
    def _paths(self, keys):
        return [self.dir / SOUND_FILES[k] for k in keys]

    def _fire(self, kind, now, sound_keys, message):
        self.last_fire[kind] = now
        self.armed[kind] = True
        self.player.play_sequence(self._paths(sound_keys))
        self.banner = message
        self.banner_until = now + BANNER_SECONDS
        print(f"[ALERT] {kind}: {message}")

    def _ready_to_repeat(self, kind, now):
        return now - self.last_fire[kind] >= REPEAT_COOLDOWN_SECONDS

    def _reset(self):
        # Recovery only clears the continuous-streak alarms. Yawning is NOT
        # reset here: yawns are momentary and the driver looks awake between
        # them, so the count must survive those gaps. It ages out on its own
        # via the 30 s sliding window and disarms when it drops below threshold.
        self.armed["sleeping"] = False
        self.armed["drowsy"] = False
        self.sleep_streak.reset()
        self.drowsy_streak.reset()

    # -- main entry point --------------------------------------------------
    def update(self, label, now=None):
        """Call once per predicted frame. Returns the current banner text."""
        if now is None:
            now = time.time()

        is_awake = label in ALERT_CLASSES

        # Recovery: back to awake long enough -> clear all timers and re-arm.
        self.awake_streak.update(is_awake, now)
        if is_awake and self.awake_streak.duration(now) >= RECOVERY_SECONDS:
            self._reset()

        # --- Sleeping: continuous for > 5 s -------------------------------
        self.sleep_streak.update(label == "Sleeping", now)
        if label == "Sleeping":
            if not self.armed["sleeping"]:
                if self.sleep_streak.duration(now) >= SLEEP_TRIGGER_SECONDS:
                    self._fire("sleeping", now, ["sleeping"],
                               "DRIVER ASLEEP - WAKE UP")
            elif self._ready_to_repeat("sleeping", now):
                self._fire("sleeping", now, ["sleeping"],
                           "STILL ASLEEP - WAKE UP")

        # --- Drowsy: continuous for > 5 s, plays two recordings -----------
        self.drowsy_streak.update(label == "Drowsy", now)
        if label == "Drowsy":
            if not self.armed["drowsy"]:
                if self.drowsy_streak.duration(now) >= DROWSY_TRIGGER_SECONDS:
                    self._fire("drowsy", now, ["drowsy_wake", "drowsy_pullover"],
                               "DROWSY - WAKE UP & PULL OVER")
            elif self._ready_to_repeat("drowsy", now):
                self._fire("drowsy", now, ["drowsy_wake", "drowsy_pullover"],
                           "STILL DROWSY - PULL OVER")

        # --- Yawning: more than 3 onsets inside the window ----------------
        if self.yawn_streak.update(label == "Yawning", now):
            self.yawn_events.append(now)  # count a new yawn onset
        while self.yawn_events and now - self.yawn_events[0] > YAWN_WINDOW_SECONDS:
            self.yawn_events.popleft()
        if len(self.yawn_events) > YAWN_TRIGGER_COUNT:
            if not self.armed["yawning"]:
                self._fire("yawning", now, ["yawning"],
                           "FREQUENT YAWNING - TAKE A REST")
            elif label == "Yawning" and self._ready_to_repeat("yawning", now):
                self._fire("yawning", now, ["yawning"],
                           "STILL YAWNING - TAKE A REST")
        else:
            # yawns aged out of the window -> allow a fresh burst to re-arm
            self.armed["yawning"] = False

        if now > self.banner_until:
            self.banner = ""
        return self.banner
