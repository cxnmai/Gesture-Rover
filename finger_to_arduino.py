import argparse
import os
import time

import cv2
import mediapipe as mp
import serial

from finger_counter import count_fingers, create_landmarker, draw_hand, ensure_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    ap.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    ap.add_argument("--num-hands", type=int, default=2, help="Max hands to detect")
    ap.add_argument(
        "--model",
        default=os.path.join("models", "hand_landmarker.task"),
        help="Path to hand_landmarker.task",
    )
    ap.add_argument(
        "--min-change-ms",
        type=int,
        default=120,
        help="Min time between serial updates (ms)",
    )
    args = ap.parse_args()

    ensure_model(args.model)
    landmarker = create_landmarker(args.model, args.num_hands)

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    time.sleep(1.5)  # allow Uno auto-reset
    ser.reset_input_buffer()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam (index {args.camera})")

    last_sent = None
    last_sent_t = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ts_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            totals = []
            if result.hand_landmarks and result.handedness:
                for idx, hand_lms in enumerate(result.hand_landmarks):
                    label = result.handedness[idx][0].category_name  # Left/Right
                    n = count_fingers(hand_lms, label)
                    totals.append((label, n))
                    draw_hand(frame, hand_lms)

            total_fingers = sum(n for _, n in totals)
            led_level = 0
            if total_fingers >= 2:
                led_level = 2
            elif total_fingers == 1:
                led_level = 1

            now = time.time()
            if (last_sent is None) or (led_level != last_sent) or ((now - last_sent_t) * 1000 >= args.min_change_ms):
                ser.write(f"{led_level}\n".encode("ascii"))
                last_sent = led_level
                last_sent_t = now

            # HUD
            cv2.putText(
                frame,
                f"Fingers: {total_fingers} -> LED level: {led_level}",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Serial: {args.port} @ {args.baud}  |  q: quit",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Finger -> Arduino", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            ser.write(b"0\n")
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
