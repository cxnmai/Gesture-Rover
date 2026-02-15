import argparse
import os
import socket
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from steering_wheel_tracker import (
    angle_deg_from_centers,
    create_landmarker,
    draw_hand,
    ensure_model,
    grip_point_px,
    hand_orientation_upright,
    wrap_deg_180,
)


def connect_tcp(host: str, port: int, timeout_s: float) -> socket.socket | None:
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.settimeout(timeout_s)
        return sock
    except OSError:
        return None


def transform_signals(reversing: bool, degree: float, hand_distance: float):
    """TODO: Customize signal transformation before transport.

    Replace this with your own mapping/scaling/quantization logic.
    Return any structure you want to encode in `build_payload`.
    """
    return {
        "reversing": reversing,
        "degree": degree,
        "hand_distance": hand_distance,
    }


def build_payload(transformed) -> bytes:
    """TODO: Convert transformed data to bytes for socket send.

    Examples you might implement:
    - fixed-size binary frame
    - compact CSV
    - protobuf/cbor/msgpack
    """
    _ = transformed
    return b""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="ESP32 IP/hostname")
    ap.add_argument("--port", type=int, default=12345, help="ESP32 TCP port (default: 12345)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    ap.add_argument(
        "--model",
        default=os.path.join("models", "hand_landmarker.task"),
        help="Path to hand_landmarker.task",
    )
    ap.add_argument("--send-ms", type=int, default=50, help="Send period in ms (default: 50)")
    ap.add_argument("--connect-timeout", type=float, default=1.0, help="TCP connect timeout seconds")
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

    smooth = deque(maxlen=7)
    prev_raw = None
    unwrap_offset = 0.0
    missing_frames = 0
    last_angle_out = 0.0
    last_angle_unwrapped = 0.0

    ori_votes = {"Left": deque(maxlen=9), "Right": deque(maxlen=9)}
    last_ori = {"Left": "unknown", "Right": "unknown"}

    sock = None
    last_connect_try = 0.0
    last_send = 0.0
    last_print = 0.0

    try:
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
                    label = result.handedness[idx][0].category_name
                    score = float(result.handedness[idx][0].score)
                    center = grip_point_px(hand_lms, w, h)

                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"

                    if label not in centers_by_label or score > centers_by_label[label][0]:
                        centers_by_label[label] = (score, center, hand_lms)

                    if not args.no_gui:
                        draw_hand(frame, hand_lms)

            for label in ("Left", "Right"):
                if label not in centers_by_label:
                    continue
                _, _, lms = centers_by_label[label]
                ori, conf = hand_orientation_upright(lms)
                if ori == "upright" and conf >= 0.35:
                    ori_votes[label].append(1)
                elif ori == "upside_down" and conf >= 0.35:
                    ori_votes[label].append(-1)
                else:
                    ori_votes[label].append(0)

                votes_sum = sum(ori_votes[label])
                if abs(votes_sum) >= 2:
                    last_ori[label] = "upright" if votes_sum > 0 else "upside_down"
                elif len(ori_votes[label]) == ori_votes[label].maxlen:
                    last_ori[label] = "unknown"

            reversing = False
            hand_distance = 0.0
            degree = last_angle_out

            if "Left" in centers_by_label and "Right" in centers_by_label:
                missing_frames = 0
                left_c = centers_by_label["Left"][1]
                right_c = centers_by_label["Right"][1]
                hand_distance = float(np.hypot(right_c[0] - left_c[0], right_c[1] - left_c[1]))

                raw = angle_deg_from_centers(left_c, right_c)

                if prev_raw is None and not smooth:
                    k = int(round((last_angle_unwrapped - raw) / 360.0))
                    unwrap_offset = 360.0 * k

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
                last_angle_unwrapped = filtered
                degree = wrap_deg_180(filtered)
                last_angle_out = degree

                reversing = (
                    abs(degree) > 160.0
                    and last_ori["Left"] == "upright"
                    and last_ori["Right"] == "upright"
                )

                if not args.no_gui:
                    cv2.line(frame, left_c, right_c, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.circle(frame, left_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, right_c, 8, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                missing_frames += 1
                if missing_frames > 10:
                    smooth.clear()
                    prev_raw = None
                    unwrap_offset = 0.0
                    ori_votes["Left"].clear()
                    ori_votes["Right"].clear()
                    last_ori["Left"] = "unknown"
                    last_ori["Right"] = "unknown"

            now = time.time()

            if sock is None and (now - last_connect_try) >= 1.0:
                last_connect_try = now
                sock = connect_tcp(args.host, args.port, args.connect_timeout)
                if sock is not None:
                    print(f"Connected to {args.host}:{args.port}")

            if (now - last_send) * 1000.0 >= args.send_ms:
                transformed = transform_signals(reversing, degree, hand_distance)
                payload = build_payload(transformed)
                if sock is not None:
                    try:
                        if payload:
                            sock.sendall(payload)
                    except OSError:
                        try:
                            sock.close()
                        except OSError:
                            pass
                        sock = None
                last_send = now

            if args.no_gui:
                if (now - last_print) > 0.25:
                    print(
                        f"reversing={reversing} degree={degree:.1f} hand_distance={hand_distance:.1f}"
                    )
                    last_print = now
                continue

            status = "CONNECTED" if sock is not None else "DISCONNECTED"
            cv2.putText(
                frame,
                f"reversing {reversing} | deg {degree:.1f} | dist {hand_distance:.1f}",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ESP {args.host}:{args.port} [{status}]  q quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture -> ESP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            if sock is not None:
                sock.close()
        except OSError:
            pass
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
