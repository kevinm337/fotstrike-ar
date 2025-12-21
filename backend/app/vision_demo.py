from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import cv2
import numpy as np

from app.vision.pipeline import VisionPipeline


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Deterministic BGR color for a given track_id.
    This keeps the same color per ID across frames.
    """
    # simple hash -> color
    r = (track_id * 37) % 255
    g = (track_id * 17) % 255
    b = (track_id * 97) % 255
    return (int(b), int(g), int(r))  # OpenCV uses BGR


def bbox_center_xyxy(bbox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return cx, cy


def draw_tracks_and_trails(
    frame_bgr: np.ndarray,
    players,
    ball,
    history: Dict[int, Deque[Tuple[int, int]]],
    trail_len: int = 25,
) -> np.ndarray:
    out = frame_bgr.copy()

    # Draw players
    for p in players:
        tid = int(p["track_id"])
        x1, y1, x2, y2 = map(int, p["bbox"])
        color = color_for_id(tid)

        # update trail history
        cx, cy = bbox_center_xyxy((x1, y1, x2, y2))
        if tid not in history:
            history[tid] = deque(maxlen=trail_len)
        history[tid].append((cx, cy))

        # draw bbox + label
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"player #{tid}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # draw trail
        pts = history[tid]
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], color, 2)

    # Draw ball (optional)
    if ball is not None:
        tid = int(ball["track_id"])
        x1, y1, x2, y2 = map(int, ball["bbox"])
        ball_color = (0, 255, 255)  # yellow
        cv2.rectangle(out, (x1, y1), (x2, y2), ball_color, 2)
        cv2.putText(
            out,
            f"ball #{tid}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ball_color,
            2,
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 12: VisionPipeline demo (boxes + IDs + trails)")
    parser.add_argument("--video-path", required=True, help="Path to input video")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--trail-len", type=int, default=25, help="How many points to keep in trails")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (1 = every frame)")
    parser.add_argument("--window-name", default="FotStrike AR - Vision Demo", help="OpenCV window title")
    parser.add_argument("--save-video", default="", help="Optional output mp4 path to save annotated demo")
    args = parser.parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read video properties for output writer (if enabled)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        out_path = Path(args.save_video).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    pipeline = VisionPipeline(conf_th=args.conf)

    # trail history: track_id -> last N centers
    history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail_len))

    frame_idx = 0
    processed = 0

    print(f"Video: {video_path.name} ({W}x{H} @ ~{fps:.2f} fps)")
    print("Controls: ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # stride: skip frames to run faster if needed
        if args.stride > 1 and (frame_idx % args.stride != 0):
            frame_idx += 1
            continue

        out = pipeline.process_frame(frame)
        players = out["players"]
        ball = out["ball"]

        annotated = draw_tracks_and_trails(
            frame_bgr=frame,
            players=players,
            ball=ball,
            history=history,
            trail_len=args.trail_len,
        )

        # small HUD text
        cv2.putText(
            annotated,
            f"frame={frame_idx} players={len(players)} ball={'yes' if ball else 'no'} conf={args.conf}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(args.window_name, annotated)

        if writer is not None:
            writer.write(annotated)

        processed += 1
        frame_idx += 1

        if args.max_frames and processed >= args.max_frames:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ Saved video: {Path(args.save_video).expanduser().resolve()}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
