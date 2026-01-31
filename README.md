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

## 3) Finger count -> Arduino LEDs

This sends the detected finger count to an Arduino Uno over Serial.

Arduino sketch:

- `arduino/finger_led_serial/finger_led_serial.ino`

Upload with Arduino CLI (example):

```bash
arduino-cli compile --fqbn arduino:avr:uno arduino/finger_led_serial
arduino-cli upload --fqbn arduino:avr:uno --port /dev/ttyACM0 arduino/finger_led_serial
```

Run the Python sender:

```bash
python finger_to_arduino.py --port /dev/ttyACM0
```

Behavior:

- 0 fingers: LED off
- 1 finger: `L` (pin 13) on
- 2+ fingers: `L` on + TX LED stays (mostly) on via periodic Serial writes

- Hold both hands like you are gripping a steering wheel.
- The program estimates rotation angle (degrees) and direction (`left`/`right`).
- It auto-calibrates once you hold the wheel level for a moment.
- Press `c` to re-calibrate (set current pose as 0 degrees).
- Press `q` to quit.
