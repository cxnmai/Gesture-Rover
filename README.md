# ECE5 Gesture Rover - CV Demos

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Finger counter

```bash
python finger_counter.py
```

Quick sanity check (no webcam / no GUI needed):

```bash
python finger_counter.py --self-test
```

- Hold up one or two hands; the overlay shows per-hand and total finger count.
- Press `q` to quit.

## 2) Imaginary steering wheel angle

```bash
python steering_wheel_tracker.py
```

Quick sanity check (no webcam / no GUI needed):

```bash
python steering_wheel_tracker.py --self-test
```

- Hold both hands like you are gripping a steering wheel.
- The program estimates rotation angle (degrees) and direction (`left`/`right`).
- It auto-calibrates once you hold the wheel level for a moment.
- Press `c` to re-calibrate (set current pose as 0 degrees).
- Press `q` to quit.
