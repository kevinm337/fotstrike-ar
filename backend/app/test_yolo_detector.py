from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.models.yolo_detector import YOLODetector


def draw_detections(frame_bgr, detections):
    # OpenCV uses BGR colors
    PLAYER_COLOR = (255, 0, 0)   # blue
    BALL_COLOR = (0, 255, 255)   # yellow

    out = frame_bgr.copy()

    for d in detections:
        label = d.label
        conf = float(d.confidence)
        x1, y1, x2, y2 = d.bbox

        # Convert bbox to ints for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        color = PLAYER_COLOR if label == "player" else BALL_COLOR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../data/matches/match1.mp4", help="Path to match video")
    parser.add_argument("--frame_idx", type=int, default=0, help="Which frame to test (0 = first frame)")

    # Separate thresholds
    parser.add_argument("--conf-player", type=float, default=0.35, help="Confidence threshold for players")
    parser.add_argument("--conf-ball", type=float, default=0.10, help="Confidence threshold for ball")

    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (can help ball)")

    parser.add_argument(
        "--out",
        default="../outputs/day5",
        help="Output folder to save annotated image",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Jump to a specific frame index
    if args.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the video.")

    detector = YOLODetector(
        model_path="yolov8n.pt",
        conf_player=args.conf_player,
        conf_ball=args.conf_ball,
        imgsz=args.imgsz,
    )
    detections = detector.detect(frame)

    print(f"Video: {video_path.name}")
    print(f"Frame index requested: {args.frame_idx}")
    print(f"Detections: {len(detections)}")

    for d in detections:
        print(f"- {d.label} conf={d.confidence:.3f} bbox={[round(x,1) for x in d.bbox]}")

    # Save annotated image
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = draw_detections(frame, detections)
    out_path = out_dir / f"yolo_test_frame_{args.frame_idx:06d}.png"
    cv2.imwrite(str(out_path), annotated)

    print(f"✅ Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
