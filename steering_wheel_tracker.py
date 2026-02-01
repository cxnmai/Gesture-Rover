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

# Points around the back-of-fist / knuckle area.
# Using MCPs (knuckles) gives a more stable "grip point" than including the wrist.
GRIP_LANDMARKS = [5, 9, 13, 17]  # index/middle/ring/pinky MCP

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


def grip_point_px(hand_landmarks, w: int, h: int) -> tuple[int, int]:
    """Approximate where the fist would grip an imaginary wheel.

    We use the knuckle (MCP) cluster and bias slightly toward the middle knuckle.
    """

    xs = [hand_landmarks[i].x for i in GRIP_LANDMARKS]
    ys = [hand_landmarks[i].y for i in GRIP_LANDMARKS]

    # Bias toward middle MCP (9) for a steadier grip point.
    x = float(np.mean(xs) * 0.65 + hand_landmarks[9].x * 0.35)
    y = float(np.mean(ys) * 0.65 + hand_landmarks[9].y * 0.35)

    cx = int(np.clip(x, 0.0, 1.0) * w)
    cy = int(np.clip(y, 0.0, 1.0) * h)
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

    # Horizontal should always mean 0 degrees (no explicit calibration).
    # We still smooth and unwrap angles to avoid jitter and +/-180 flipping.
    smooth = deque(maxlen=7)
    # Hand separation tracking: calibrated to user's "neutral" wheel width.
    dist_filter = deque(maxlen=7)
    dist_calib = deque(maxlen=15)
    baseline_dist = None
    prev_raw = None
    unwrap_offset = 0.0
    missing_frames = 0
    last_print = 0.0
    last_angle_out = 0.0

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
                label = result.handedness[idx][0].category_name  # Left/Right (image)
                score = float(result.handedness[idx][0].score)
                center = grip_point_px(hand_lms, w, h)

                # We process a mirrored frame for user-friendly display.
                # Mirroring swaps image-handedness, so flip labels back.
                if label == "Left":
                    label = "Right"
                elif label == "Right":
                    label = "Left"

                # Keep the most confident instance per label.
                if label not in centers_by_label or score > centers_by_label[label][0]:
                    centers_by_label[label] = (score, center)

                if not args.no_gui:
                    draw_hand(frame, hand_lms)

        msg = "Show two hands like a steering wheel"
        angle_out = None
        dist_px = 0.0
        dist_ratio = 0.0
        dist_note = ""

        if "Left" in centers_by_label and "Right" in centers_by_label:
            missing_frames = 0
            # Use per-hand identity so the angle can exceed 90 degrees.
            left_c = centers_by_label["Left"][1]
            right_c = centers_by_label["Right"][1]

            dist_px_raw = float(math.hypot(right_c[0] - left_c[0], right_c[1] - left_c[1]))
            dist_filter.append(dist_px_raw)
            dist_px = float(np.mean(dist_filter))

            if baseline_dist is None:
                dist_calib.append(dist_px)
                if len(dist_calib) == dist_calib.maxlen:
                    baseline_dist = float(np.mean(dist_calib))
                else:
                    dist_note = "(calibrating width)"
            if baseline_dist is not None and baseline_dist > 1e-6:
                dist_ratio = dist_px / baseline_dist

            raw = angle_deg_from_centers(left_c, right_c)  # (-180, 180]

            # If we previously lost tracking and cleared unwrap state, align the
            # unwrap branch so the angle stays continuous when hands reappear.
            if prev_raw is None and not smooth:
                k = int(round((last_angle_out - raw) / 360.0))
                unwrap_offset = 360.0 * k

            # Unwrap to allow continuous rotation past +/-180.
            if prev_raw is not None:
                diff = raw - prev_raw
                if diff > 180.0:
                    unwrap_offset -= 360.0
                elif diff < -180.0:
                    unwrap_offset += 360.0
            prev_raw = raw

            unwrapped = raw + unwrap_offset

            smooth.append(unwrapped)
            filtered = float(np.mean(smooth))
            angle_out = filtered
            last_angle_out = float(angle_out)

            direction = "center"
            if angle_out > 3:
                direction = "right"
            elif angle_out < -3:
                direction = "left"

            msg = f"Steering: {abs(angle_out):.1f} deg {direction}"
            if baseline_dist is not None:
                msg += f" | Width: {dist_ratio:.2f}x"
            else:
                msg += f" | Width: -- {dist_note}".rstrip()

            if not args.no_gui:
                cv2.line(frame, left_c, right_c, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.circle(frame, left_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, right_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "L grip",
                    (left_c[0] - 30, left_c[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "R grip",
                    (right_c[0] - 30, right_c[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            missing_frames += 1
            if missing_frames > 10:
                smooth.clear()
                dist_filter.clear()
                prev_raw = None
                unwrap_offset = 0.0

            # Hold the last known steering angle when hands are not visible.
            hold_dir = "center"
            if last_angle_out > 3:
                hold_dir = "right"
            elif last_angle_out < -3:
                hold_dir = "left"
            msg = f"Steering: {abs(last_angle_out):.1f} deg {hold_dir} | Width: 0"

        if args.no_gui:
            now = time.time()
            if (now - last_print) > 0.25:
                if angle_out is None:
                    print(f"Steering: {last_angle_out:.1f} deg | Width: 0")
                elif baseline_dist is None:
                    print(f"Steering: {angle_out:.1f} deg | Width: --")
                else:
                    print(f"Steering: {angle_out:.1f} deg | Width: {dist_ratio:.2f}x")
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

        cv2.putText(
            frame,
            "q: quit",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Steering wheel overlay: hold last angle when hands not detected.
        wheel_angle = float(last_angle_out) if angle_out is None else float(angle_out)
        wheel_center = (frame.shape[1] - 150, frame.shape[0] - 150)
        draw_steering_wheel(frame, wheel_center, radius=110, angle_deg=wheel_angle)

        # Distance HUD
        if baseline_dist is None:
            dist_line = f"Wheel width: calibrating ({len(dist_calib)}/{dist_calib.maxlen})"
        else:
            dist_line = f"Wheel width: {dist_ratio:.2f}x  ({dist_px:.0f}px)"
        if angle_out is None:
            dist_line = "Wheel width: 0"
        cv2.putText(
            frame,
            dist_line,
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "r: recal width",
            (10, frame.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Steering Wheel Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            baseline_dist = None
            dist_calib.clear()
            dist_filter.clear()

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
