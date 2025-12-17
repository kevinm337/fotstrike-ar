from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.models.yolo_detector import YOLODetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True, help="Path to input video (mp4/mov/mkv)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = process entire video, else stop after N frames")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for YOLO detections")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (1 = all frames, 2 = every other)")
    args = parser.parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    detector = YOLODetector(model_path="yolov8n.pt", conf_th=args.conf)

    total_frames_read = 0
    processed_frames = 0

    players_sum = 0
    ball_frames = 0

    max_frames = None if args.max_frames == 0 else args.max_frames
    stride = max(1, args.stride)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        total_frames_read += 1

        # stride: only process every Nth frame
        if (total_frames_read - 1) % stride != 0:
            continue

        detections = detector.detect(frame)

        num_players = sum(1 for d in detections if d.label == "player")
        has_ball = any(d.label == "ball" for d in detections)


        players_sum += num_players
        if has_ball:
            ball_frames += 1

        processed_frames += 1

        if processed_frames % 20 == 0:
            print(
                f"Processed {processed_frames} frames "
                f"(read={total_frames_read}) | players={num_players} | ball={'yes' if has_ball else 'no'}"
            )

        if max_frames is not None and processed_frames >= max_frames:
            break

    cap.release()

    if processed_frames == 0:
        print("No frames processed. Check video-path / max-frames / stride.")
        return

    avg_players = players_sum / processed_frames
    ball_pct = (ball_frames / processed_frames) * 100.0

    print("\n===== YOLO Summary =====")
    print(f"Video: {video_path.name}")
    print(f"Frames read: {total_frames_read}")
    print(f"Frames processed: {processed_frames} (stride={stride})")
    print(f"Confidence threshold: {args.conf}")
    print(f"Average players per frame: {avg_players:.2f}")
    print(f"Ball detected in: {ball_frames}/{processed_frames} frames ({ball_pct:.1f}%)")


if __name__ == "__main__":
    main()
