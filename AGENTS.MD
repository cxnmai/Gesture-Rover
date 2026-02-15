# ECE5_GestureRover Codebase Guide (AGENTS.MD)

This repo currently centers on camera-based steering extraction and packet generation for a differential-drive rover.

## Current Project Focus

Primary workflow:

1. `comp_vision_input/steering_wheel_tracker.py`
   - Tracks two hands as a steering wheel.
   - Computes steering angle, width scalar, and reversing flag.
   - Shows live packet preview and interpreted wheel motion (what the car would do).

2. `comp_vision_input/cv_to_packet.py`
   - Converts `width + angle` into left/right wheel commands.
   - Encodes packet format from `plan.md`: `RLSL RRSR` as 8 digits:
     - `ReverseLeft` (1 digit, `0/1`)
     - `SpeedLeft` (3 digits, `000..255`)
     - `ReverseRight` (1 digit, `0/1`)
     - `SpeedRight` (3 digits, `000..255`)

3. `comp_vision_input/gesture_to_esp.py`
   - Tracks the same steering signals and connects to ESP over TCP.
   - Contains pluggable hooks (`transform_signals`, `build_payload`) for custom payload format.
   - By default, `build_payload` returns empty bytes and sends nothing until customized.

4. `esp32/wifi_connect.ino`
   - ESP32 Wi-Fi TCP server on port `12345`.
   - Parses fixed-size frames (`<Bhh>`) and logs decoded values.
   - This parser is currently separate from the packet format in `cv_to_packet.py`.

## Repository Layout

- `comp_vision_input/steering_wheel_tracker.py`
  - Main visual tracker + packet/motion preview overlay.

- `comp_vision_input/cv_to_packet.py`
  - Differential-drive math and packet encoding.

- `comp_vision_input/gesture_to_esp.py`
  - Network bridge framework (custom payload hooks).

- `comp_vision_input/models/hand_landmarker.task`
  - MediaPipe model artifact (local copy).

- `esp32/wifi_connect.ino`
  - ESP32 receiver sketch.

- `plan.md`
  - Authoritative packet field format for rover control packets.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run steering tracker with live packet preview:

```bash
python comp_vision_input/steering_wheel_tracker.py
```

Run packet conversion directly:

```bash
python comp_vision_input/cv_to_packet.py --width 1.1 --angle -35
```

Run ESP bridge framework:

```bash
python comp_vision_input/gesture_to_esp.py --host <ESP_IP> --port 12345
```

## Steering + Packet Math (Current Behavior)

In `comp_vision_input/cv_to_packet.py`:

- Speed source is width scalar (`w` from tracker).
- Speed thresholds:
  - `w < 0.2` -> speed `0` (braking zone).
  - `w > 1.7` -> treated as `1.7` (top-speed cap).
- Turning uses differential wheel mix from steering angle.
- Turning remains possible while stationary (pivot behavior).
- If one side would clip above `255`, both sides are scaled together to preserve turn ratio.
- Steering convention:
  - negative angle (left turn) -> right wheel faster
  - positive angle (right turn) -> left wheel faster

## Width Scaling in Tracker

`comp_vision_input/steering_wheel_tracker.py` uses screen-space scaling:

- Width ratio is computed from hand distance in pixels relative to frame width.
- Near full-frame hand span displays near `2.00x`.
- This is intentionally screen-referenced and not calibration/depth-normalized.

## Where To Change What

- Change packet mapping and motor math:
  - `comp_vision_input/cv_to_packet.py`

- Change what tracker displays/sends:
  - `comp_vision_input/steering_wheel_tracker.py`
  - `comp_vision_input/gesture_to_esp.py`

- Change ESP network receiver/parser:
  - `esp32/wifi_connect.ino`

## Troubleshooting

- Camera open failure:
  - Try `--camera 1`.
  - Ensure `/dev/video*` permissions and no camera lock by another app.

- Model file issues:
  - Ensure `hand_landmarker.task` exists where script expects it.

- Import path issues:
  - Run scripts from repository root so sibling imports resolve.

- Protocol mismatch warning:
  - `cv_to_packet.py` emits 8-digit ASCII control packets.
  - `wifi_connect.ino` currently parses binary `<Bhh>` frames.
  - Align these before end-to-end testing.
