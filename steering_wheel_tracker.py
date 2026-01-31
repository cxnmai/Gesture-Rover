import argparse
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

PALM_LANDMARKS = [0, 5, 9, 13, 17]  # wrist + MCPs

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model(model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return
    print(f"Downloading model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def create_landmarker(model_path: str, num_hands: int):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand(frame, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    pts = []
    for lm in hand_landmarks:
        x = int(np.clip(lm.x, 0.0, 1.0) * w)
        y = int(np.clip(lm.y, 0.0, 1.0) * h)
        pts.append((x, y))

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)


def palm_center_px(hand_landmarks, w: int, h: int) -> tuple[int, int]:
    xs = [hand_landmarks[i].x for i in PALM_LANDMARKS]
    ys = [hand_landmarks[i].y for i in PALM_LANDMARKS]
    cx = int(np.clip(float(np.mean(xs)), 0.0, 1.0) * w)
    cy = int(np.clip(float(np.mean(ys)), 0.0, 1.0) * h)
    return cx, cy


def angle_deg_from_centers(left: tuple[int, int], right: tuple[int, int]) -> float:
    """Angle of line from left->right in degrees (image coordinates).

    0 deg: horizontal.
    Positive: right side lower (clockwise / steering right).
    Negative: right side higher (counterclockwise / steering left).
    """

    (x1, y1), (x2, y2) = left, right
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def draw_steering_wheel(
    frame,
    center: tuple[int, int],
    radius: int,
    angle_deg: float,
    alpha: float = 0.65,
) -> None:
    """Draw a simple steering wheel overlay rotated by angle_deg."""

    h, w = frame.shape[:2]
    cx, cy = center
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    radius = int(max(10, radius))

    overlay = frame.copy()
    rim_color = (30, 30, 30)
    rim_hi = (245, 245, 245)
    spoke_color = (220, 220, 220)
    hub_color = (60, 60, 60)

    # Outer/inner rim
    cv2.circle(overlay, (cx, cy), radius, rim_hi, 10, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), radius, rim_color, 6, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), int(radius * 0.72), (0, 0, 0), -1, cv2.LINE_AA)

    # Spokes (3) rotated by angle
    theta0 = math.radians(angle_deg)
    for k in range(3):
        theta = theta0 + k * (2.0 * math.pi / 3.0)
        x2 = int(cx + math.cos(theta) * radius * 0.9)
        y2 = int(cy + math.sin(theta) * radius * 0.9)
        cv2.line(overlay, (cx, cy), (x2, y2), spoke_color, 6, cv2.LINE_AA)

    # Hub
    cv2.circle(overlay, (cx, cy), int(radius * 0.18), hub_color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), int(radius * 0.18), (255, 255, 255), 2, cv2.LINE_AA)

    # Small top marker (helps visualize direction)
    tx = int(cx + math.cos(theta0 - math.pi / 2.0) * radius * 0.95)
    ty = int(cy + math.sin(theta0 - math.pi / 2.0) * radius * 0.95)
    cv2.circle(overlay, (tx, ty), 6, (0, 255, 255), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    ap.add_argument(
        "--model",
        default=os.path.join("models", "hand_landmarker.task"),
        help="Path to hand_landmarker.task",
    )
    ap.add_argument("--no-gui", action="store_true", help="Run without cv2.imshow")
    ap.add_argument("--self-test", action="store_true", help="Load model and exit")
    args = ap.parse_args()

    if os.environ.get("DISPLAY") is None:
        args.no_gui = True

    ensure_model(args.model)
    landmarker = create_landmarker(args.model, num_hands=2)

    if args.self_test:
        print("OK: model loaded")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam (index {args.camera})")

    baseline = None  # unwrapped angle baseline
    smooth = deque(maxlen=7)
    prev_raw = None
    unwrap_offset = 0.0
    missing_frames = 0
    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        centers_by_label = {}
        if result.hand_landmarks and result.handedness:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                label = result.handedness[idx][0].category_name  # Left/Right
                score = result.handedness[idx][0].score
                center = palm_center_px(hand_lms, w, h)

                # Keep the most confident instance per label.
                if label not in centers_by_label or score > centers_by_label[label][0]:
                    centers_by_label[label] = (score, center)

                if not args.no_gui:
                    draw_hand(frame, hand_lms)

        msg = "Show two hands like a steering wheel"
        angle_out = None

        if "Left" in centers_by_label and "Right" in centers_by_label:
            missing_frames = 0
            left_c = centers_by_label["Left"][1]
            right_c = centers_by_label["Right"][1]

            raw = angle_deg_from_centers(left_c, right_c)  # (-180, 180]

            # Unwrap to allow continuous rotation past +/-180.
            if prev_raw is not None:
                diff = raw - prev_raw
                if diff > 180.0:
                    unwrap_offset -= 360.0
                elif diff < -180.0:
                    unwrap_offset += 360.0
            prev_raw = raw

            unwrapped = raw + unwrap_offset

            if baseline is None:
                smooth.append(unwrapped)
                if len(smooth) == smooth.maxlen:
                    baseline = float(np.mean(smooth))
            else:
                smooth.append(unwrapped)

            if baseline is not None and smooth:
                filtered = float(np.mean(smooth))
                angle_out = float(filtered - baseline)

                direction = "center"
                if angle_out > 3:
                    direction = "right"
                elif angle_out < -3:
                    direction = "left"

                msg = f"Steering: {abs(angle_out):.1f} deg {direction}"

            if not args.no_gui:
                cv2.line(frame, left_c, right_c, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.circle(frame, left_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, right_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            missing_frames += 1
            if missing_frames > 30:
                # If tracking is lost for a bit, reset unwrap state so we don't
                # accumulate a large, incorrect offset when hands reappear.
                prev_raw = None
                unwrap_offset = 0.0

        if args.no_gui:
            now = time.time()
            if baseline is None:
                if (now - last_print) > 1.0:
                    print("Calibrating... hold wheel level")
                    last_print = now
            else:
                if angle_out is not None and (now - last_print) > 0.25:
                    print(msg)
                    last_print = now
            continue

        cv2.putText(
            frame,
            msg,
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255) if angle_out is None else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if baseline is None:
            cv2.putText(
                frame,
                "Calibrating... hold wheel level",
                (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "c: re-calibrate | q: quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Steering wheel overlay (always draw; angle 0 while calibrating)
        wheel_angle = 0.0 if angle_out is None else float(angle_out)
        wheel_center = (frame.shape[1] - 150, frame.shape[0] - 150)
        draw_steering_wheel(frame, wheel_center, radius=110, angle_deg=wheel_angle)

        cv2.imshow("Steering Wheel Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and ("Left" in centers_by_label and "Right" in centers_by_label):
            left_c = centers_by_label["Left"][1]
            right_c = centers_by_label["Right"][1]
            raw = angle_deg_from_centers(left_c, right_c)

            # Update unwrap state then set baseline on the unwrapped value.
            if prev_raw is not None:
                diff = raw - prev_raw
                if diff > 180.0:
                    unwrap_offset -= 360.0
                elif diff < -180.0:
                    unwrap_offset += 360.0
            prev_raw = raw
            baseline = float(raw + unwrap_offset)
            smooth.clear()

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
