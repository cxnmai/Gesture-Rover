import argparse
import os
import time
import urllib.request

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
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def count_fingers(hand_landmarks, handedness_label: str) -> int:
    """Return number of raised fingers for a single hand.

    Landmarks are MediaPipe normalized landmarks (0..1).
    - Index/Middle/Ring/Pinky: tip above PIP (smaller y)
    - Thumb: tip away from palm along x depending on handedness
    """

    lm = hand_landmarks

    thumb_tip, thumb_ip = 4, 3
    index_tip, index_pip = 8, 6
    middle_tip, middle_pip = 12, 10
    ring_tip, ring_pip = 16, 14
    pinky_tip, pinky_pip = 20, 18

    fingers_up = 0

    if handedness_label.lower() == "right":
        if lm[thumb_tip].x < lm[thumb_ip].x:
            fingers_up += 1
    else:
        if lm[thumb_tip].x > lm[thumb_ip].x:
            fingers_up += 1

    for tip, pip in (
        (index_tip, index_pip),
        (middle_tip, middle_pip),
        (ring_tip, ring_pip),
        (pinky_tip, pinky_pip),
    ):
        if lm[tip].y < lm[pip].y:
            fingers_up += 1

    return fingers_up


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    ap.add_argument("--num-hands", type=int, default=2, help="Max hands to detect")
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
    landmarker = create_landmarker(args.model, args.num_hands)

    if args.self_test:
        print("OK: model loaded")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam (index {args.camera})")

    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        totals = []
        if result.hand_landmarks and result.handedness:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                label = result.handedness[idx][0].category_name  # Left/Right
                n = count_fingers(hand_lms, label)
                totals.append((label, n))
                draw_hand(frame, hand_lms)

        if args.no_gui:
            now = time.time()
            if totals and (now - last_print) > 0.25:
                total_count = sum(n for _, n in totals)
                print(f"Total fingers: {total_count} | " + ", ".join(f"{l}:{n}" for l, n in totals))
                last_print = now
            continue

        y = 30
        if totals:
            total_count = sum(n for _, n in totals)
            cv2.putText(
                frame,
                f"Total fingers: {total_count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 32
            for label, n in totals:
                cv2.putText(
                    frame,
                    f"{label} hand: {n}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28
        else:
            cv2.putText(
                frame,
                "Show hand(s) to camera",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
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

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
