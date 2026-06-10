# Alert recordings

Put your alarm recordings here as **.wav** files with these exact names.
Until a file exists, the system plays a plain beep instead, so the demo still
works before you record the real messages.

| File name              | When it plays                                   | Suggested content                                  |
|------------------------|-------------------------------------------------|----------------------------------------------------|
| `sleeping_alert.wav`   | Driver asleep > 5 s continuously                | A loud wake-up alarm / "Wake up!"                  |
| `drowsy_wake.wav`      | Head nodding (Drowsy) > 5 s — **plays first**   | "Wake up, stay alert!"                             |
| `drowsy_pullover.wav`  | Right after `drowsy_wake.wav` — **plays second**| "You seem tired, pull over to refresh."           |
| `yawning_alert.wav`    | More than 3 yawns within 30 s                   | "You're yawning a lot, consider stopping to rest." |

## The rules (defined in `src/alerts.py`)

- **Sleeping** — asleep continuously for more than 5 seconds → `sleeping_alert.wav`.
- **Drowsy** — nodding continuously for more than 5 seconds → `drowsy_wake.wav`
  then `drowsy_pullover.wav`, one after the other.
- **Yawning** — more than 3 yawns inside a 30-second window → `yawning_alert.wav`.
- **After an alert**, while the same danger state keeps happening, the matching
  recording repeats (ignoring the time threshold) with an 8-second cooldown.
- When the driver is **Alert / Singing** again for ~1.5 seconds, everything
  resets and the alarms can arm again.

To change any timing or threshold, edit the constants at the top of
`src/alerts.py` (e.g. `SLEEP_TRIGGER_SECONDS`, `YAWN_TRIGGER_COUNT`).

The class groups (which states count as danger vs awake) live in
`src/config.py`: `DANGER_CLASSES`, `ALERT_CLASSES`, `NEUTRAL_CLASSES`.
