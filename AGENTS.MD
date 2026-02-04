# ECE5_GestureRover Codebase Guide (AGENTS.MD)

This repo is a small, single-purpose prototype for controlling/visualizing "rover" interactions using hand tracking.
It is primarily:

- Python + OpenCV + MediaPipe Tasks for webcam hand landmark detection.
- Optional serial output to an Arduino Uno sketch to drive LEDs as a simple hardware feedback channel.

The code is intentionally script-first (no packages, no tests) and is optimized for quick iteration in a lab setting.

## What This Project Does

There are three main runtime modes (three Python entrypoints):

1. `finger_counter.py`
   - Detects up to N hands and counts "raised fingers" per hand.
   - Shows an overlay (skeleton + text HUD) or prints to stdout in headless mode.

2. `steering_wheel_tracker.py`
   - Uses two hands held like a steering wheel.
   - Computes an angle based on the left/right "grip points" and smooths/unwraps it.
   - Estimates "wheel width" (distance between hands) relative to a calibrated baseline.
   - Draws a steering wheel overlay that rotates with the measured angle.

3. `finger_to_arduino.py`
   - Runs the same finger counting as `finger_counter.py`.
   - Maps total finger count to a discrete `LED level` and streams it to an Arduino over serial.

The Arduino sketch `arduino/finger_led_serial/finger_led_serial.ino` listens for newline-terminated integers and drives the onboard LED. A "level 2" mode also keeps the TX LED lit by periodically transmitting bytes.

## Repository Layout

Key files/directories:

- `requirements.txt`
  - Python deps: OpenCV, MediaPipe, NumPy, PySerial.

- `finger_counter.py`
  - Finger counting demo + reusable functions.

- `steering_wheel_tracker.py`
  - Two-hand steering angle + wheel overlay.

- `finger_to_arduino.py`
  - Finger count -> serial control.

- `models/hand_landmarker.task`
  - MediaPipe hand landmark model asset (binary). This is large.
  - Note: `models/*.task` is ignored by `.gitignore`. The scripts can download the model if missing.

- `arduino/finger_led_serial/finger_led_serial.ino`
  - Arduino Uno serial-controlled LED behavior.

- `models/uno_led_test.ino`
  - A simple LED blink test sketch (also writes to serial).

Also present:

- `.venv/` is commonly used locally; it is ignored.
- `__pycache__/` is ignored.

## Quick Start (Python)

Create a venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run finger counting:

```bash
python finger_counter.py
```

Run steering tracking:

```bash
python steering_wheel_tracker.py
```

Run finger -> Arduino serial control:

```bash
python finger_to_arduino.py --port /dev/ttyACM0
```

Notes:

- All scripts default to webcam index `--camera 0`.
- On systems without a GUI (`DISPLAY` unset), the Python scripts automatically switch to `--no-gui` behavior (stdout logs only).
- Each Python entrypoint supports `--self-test` to load the model and exit.

## Model Asset Handling

The Python scripts use MediaPipe "Tasks" and require a hand landmark model file:

- Default path: `models/hand_landmarker.task`
- If missing, `ensure_model()` downloads it from:
  - `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`

Implementation details:

- `ensure_model(model_path)` creates parent directories and downloads only if the file does not exist.
- The model is treated like a cacheable artifact; it is not meant to be checked into git (and is ignored by `.gitignore`).

If you want a deterministic build, pin the exact model file/version yourself (e.g., store it externally and copy into `models/`).

## Core Pipeline (Shared)

All three Python scripts share the same core flow:

1. Open webcam: `cv2.VideoCapture(camera_index)`
2. Read frame -> mirror for user-friendly display: `cv2.flip(frame, 1)`
3. Convert BGR -> RGB: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
4. Wrap into a MediaPipe image: `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`
5. Call `detect_for_video(mp_image, ts_ms)` where `ts_ms = int(time.time() * 1000)`
6. Use `result.hand_landmarks` and `result.handedness` (if present)

This is designed for real-time use and uses `vision.RunningMode.VIDEO`.

## `finger_counter.py`

This script acts as both a demo and a utility module.

Primary responsibilities:

- Model download and initialization:
  - `ensure_model(model_path)`
  - `create_landmarker(model_path, num_hands)`

- Finger classification:
  - `count_fingers(hand_landmarks, handedness_label)` returns an integer 0..5.
  - Rules used:
    - Index/Middle/Ring/Pinky: finger is "up" if `tip.y < pip.y` in normalized coordinates.
    - Thumb: uses x-axis comparison and depends on handedness label:
      - Right hand: thumb "up" if `tip.x < ip.x`
      - Left hand: thumb "up" if `tip.x > ip.x`

- Visualization:
  - `draw_hand(frame, hand_landmarks)` draws a simplified skeleton using a custom `HAND_CONNECTIONS` list.
  - This avoids MediaPipe drawing utils and keeps OpenCV-only drawing.

CLI flags:

- `--camera <int>`
- `--num-hands <int>`
- `--model <path>`
- `--no-gui`
- `--self-test`

Headless behavior:

- In `--no-gui` mode, it prints a line about every ~250ms when hands are visible:
  - `Total fingers: <sum> | Left:<n>, Right:<n>` (hand labels come from MediaPipe handedness).

## `finger_to_arduino.py`

This script is the "bridge" between computer vision and microcontroller output.

How it reuses code:

- Imports from `finger_counter.py`:
  - `ensure_model`, `create_landmarker`, `draw_hand`, `count_fingers`

Serial protocol (PC -> Arduino):

- Opens a serial port using PySerial:
  - Default: `--port /dev/ttyACM0`
  - Default baud: `--baud 115200`
- Sends ASCII text lines: `"<level>\n"` where `level` is an integer 0, 1, or 2.
- The script rate-limits updates with `--min-change-ms` (default 120ms) and also sends on any value change.

Mapping from fingers to LED level:

- `0 fingers` -> `0`
- `1 finger` -> `1`
- `>=2 fingers` -> `2`

Shutdown safety:

- In `finally`, attempts to send `0` before closing serial.

If you later add rover movement control, this is the file most likely to evolve into a "control loop" process.

## `steering_wheel_tracker.py`

This script implements "two hands = steering wheel" tracking.

Key ideas:

- Grip point definition:
  - Uses knuckle landmarks (`MCP` joints) instead of the wrist for stability.
  - `GRIP_LANDMARKS = [5, 9, 13, 17]` (index/middle/ring/pinky MCP).
  - `grip_point_px()` averages these and biases toward landmark 9 (middle MCP).

- Mirroring and handedness:
  - The frame is flipped with `cv2.flip(frame, 1)`.
  - MediaPipe handedness labels are image-based; flipping swaps left/right.
  - The code explicitly flips labels back:
    - If MediaPipe says `Left`, treat it as `Right`, and vice versa.
  - Then it keeps the most confident detection per label.

- Steering angle:
  - The steering "raw" angle is computed from the line `left_grip -> right_grip`:
    - `angle_deg_from_centers(left, right) = degrees(atan2(dy, dx))`
  - Interpretation:
    - `0 deg` means hands are horizontally aligned.
    - Positive means the right hand is lower in the image (clockwise rotation / "steer right").

- Displayed angle is wrapped:
  - Internally the angle is unwrapped/smoothed for continuity.
  - The value shown in the UI/logs is wrapped to `[-180, 180)` so it stays pinned around 0 and does not show `360`.

- Jitter control:
  - A short moving average smooths the angle (`deque(maxlen=7)`).
  - Angle unwrapping avoids jumps near +/-180 degrees.
  - When tracking is lost for >10 frames, smoothing state is cleared.
  - When tracking is lost, it "holds" the last known angle to keep the overlay stable.

- Wheel width estimation:
  - Distance between grips is computed in pixels and smoothed.
  - The first ~15 frames establish `baseline_dist` ("neutral" wheel width).
  - The UI reports `dist_ratio = dist_px / baseline_dist` as a compact `w <ratio>x`.
  - Press `r` to recalibrate baseline width.

- Hand orientation + reversing:
  - Each hand is classified as `upright`, `upside_down`, or `unknown` using the wrist -> middle-fingertip direction.
  - The classification is smoothed with a small vote window per hand.
  - `reversing` becomes `True` when:
    - `abs(deg) > 160` AND both hands are `upright`.

- Overlay rendering:
  - `draw_steering_wheel()` draws a rim + spokes + hub into an overlay and alpha-blends it.
  - A small top marker helps indicate rotation direction.
  - The overlay uses `last_angle_out` if hands disappear.

- Minimal UI text:
  - Main HUD line: `deg <value> | w <value> | reversing <True/False>`
  - Width format:
    - `w cal <n>/<max>` during calibration
    - `w <ratio>x` after calibration
    - `w 0` when hands are missing

CLI flags:

- `--camera <int>`
- `--model <path>`
- `--no-gui`
- `--self-test`

If you want to drive a rover, this script is the natural place to transform `angle_out` (and possibly width) into a steering command.

## Arduino Side

### `arduino/finger_led_serial/finger_led_serial.ino`

Behavior:

- Reads newline-terminated integer lines from Serial.
- Clamps values into `0..2`.
- Fail-safe: if no valid command arrives for ~1200ms, it turns the LED off.
- `level >= 1`: turns `LED_BUILTIN` on.
- `level >= 2`: transmits a byte every ~20ms to keep the TX LED "on" (approx).

Protocol expectations:

- Baud rate: `115200`.
- Messages: ASCII integer + `\n` (or `\r\n`).
- The sketch tolerates extra whitespace/newlines and ignores empty lines.

### `models/uno_led_test.ino`

This is a standalone sanity test:

- Blinks `LED_BUILTIN` every 500ms.
- Writes bytes to Serial to make the TX LED flicker.

## Operational Notes / Troubleshooting

Webcam:

- If `Could not open webcam`, try:
  - A different index: `--camera 1`
  - Closing other apps using the camera.
  - Checking Linux device permissions on `/dev/video*`.

MediaPipe model:

- If you see download issues:
  - Ensure outbound network access.
  - Manually place the model at `models/hand_landmarker.task`.

Serial (Linux):

- If you get `permission denied` on `/dev/ttyACM0`:
  - Add your user to `dialout` (or the distro's serial group) and re-login.

Arduino auto-reset:

- `finger_to_arduino.py` sleeps ~1.5s after opening the port to allow the Uno to reset.

Headless runs:

- If running over SSH without X forwarding, `DISPLAY` will likely be unset and the scripts will automatically switch to `--no-gui`.

## Conventions and Design Choices

- Script-first structure: functions live in top-level `.py` files.
- Minimal dependencies: no web frameworks, no packaging.
- Avoids MediaPipe drawing helpers and instead draws directly via OpenCV for control and portability.
- Uses `vision.RunningMode.VIDEO` + `detect_for_video()` with a millisecond timestamp.

## Where To Make Common Changes

- Change finger counting thresholds/logic:
  - `finger_counter.py` (`count_fingers()`)

- Add a new output mapping (e.g., speed based on finger count):
  - `finger_to_arduino.py` (the `led_level` mapping section)

- Map steering to rover control:
  - `steering_wheel_tracker.py` (after `angle_out` is computed)
  - Suggested pattern: normalize angle into a bounded range (e.g., -1..1) and send over serial.

- Change serial protocol:
  - PC side: `finger_to_arduino.py` (`ser.write(...)`)
  - Arduino side: `arduino/finger_led_serial/finger_led_serial.ino` (`readLevelLine()`)

## Suggested Extensions (If You're Evolving This Repo)

- Consolidate shared helpers:
  - `ensure_model`, `create_landmarker`, `draw_hand` are duplicated between scripts.
  - Consider a small module (e.g., `vision_utils.py`) if you add more modes.

- Add a "control protocol" version:
  - If you move beyond a single integer, define a line-based protocol:
    - Example: `STEER <deg> WIDTH <ratio>`
  - Keep Arduino parsing robust (overflow handling already exists).

- Add basic tests:
  - Unit-test `count_fingers()` with synthetic landmarks.
  - Unit-test angle unwrapping logic in `steering_wheel_tracker.py`.

- Add a simple `Makefile` or `justfile`:
  - For repeatable `venv` creation and common run commands.

## Privacy / Safety

- The scripts access a live camera stream. No explicit recording/upload is implemented.
- If you add logging/recording, document data retention and get consent in your environment.
