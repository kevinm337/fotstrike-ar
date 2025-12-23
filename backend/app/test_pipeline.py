from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.vision.pipeline import VisionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../data/matches/match1.mp4", help="Path to match video")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to test")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if args.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from video.")

    pipeline = VisionPipeline(model_path="yolov8n.pt", conf_th=args.conf)
    out = pipeline.process_frame(frame)

    players = out["players"]
    ball = out["ball"]

    print(f"Video: {video_path.name}")
    print(f"Frame: {args.frame_idx}")
    print(f"Players tracked: {len(players)}")
    if ball is None:
        print("Ball: None")
    else:
        print(f"Ball: id={ball['track_id']} bbox={[round(x,1) for x in ball['bbox']]}")

    # Print first few players
    for p in players[:8]:
        print(f"- player id={p['track_id']} bbox={[round(x,1) for x in p['bbox']]}")

if __name__ == "__main__":
    main()
