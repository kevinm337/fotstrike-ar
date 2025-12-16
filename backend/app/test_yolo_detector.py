from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.models.yolo_detector import YOLODetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../data/matches/match1.mp4", help="Path to match video")
    parser.add_argument("--frame_idx", type=int, default=0, help="Which frame to test (0 = first frame)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Jump to a specific frame index (optional)
    if args.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the video.")

    detector = YOLODetector(model_path="yolov8n.pt", conf_th=args.conf)
    detections = detector.detect(frame)

    print(f"Video: {video_path.name}")
    print(f"Frame index requested: {args.frame_idx}")
    print(f"Detections: {len(detections)}")
    for d in detections:
        # pretty print
        label = d["label"]
        conf = d["confidence"]
        bbox = d["bbox"]
        print(f"- {label} conf={conf:.3f} bbox={[round(x,1) for x in bbox]}")


if __name__ == "__main__":
    main()
