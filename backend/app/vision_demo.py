from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from app.vision.pipeline import VisionPipeline
from app.analytics.state import MatchState


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    x = (track_id * 37) % 255
    y = (track_id * 17) % 255
    z = (track_id * 97) % 255
    return (int(x), int(y), int(z))  # BGR


def bbox_center_xyxy(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return cx, cy


def draw_tracks_with_stats(
    frame_bgr,
    players: List[dict],
    ball: Optional[dict],
    match_state: MatchState,
    trail_history: Dict[int, List[Tuple[int, int]]],
    trail_len: int = 25,
) -> None:
    # ---------- draw players ----------
    for p in players:
        if p.get("label") != "player":
            continue

        tid = int(p["track_id"])
        x1, y1, x2, y2 = map(int, p["bbox"])
        color = color_for_id(tid)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        st = match_state.players.get(tid)
        if st is not None:
            speed = float(st.current_speed)
            dist = float(st.total_distance)
            text = f"ID {tid} | {speed:.2f} m/s | {dist:.1f} m"
        else:
            text = f"ID {tid}"

        cv2.putText(
            frame_bgr,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        cx, cy = bbox_center_xyxy([x1, y1, x2, y2])
        trail_history[tid].append((cx, cy))
        trail_history[tid] = trail_history[tid][-trail_len:]

        pts = trail_history[tid]
        for i in range(1, len(pts)):
            cv2.line(frame_bgr, pts[i - 1], pts[i], color, 2)

    # ---------- draw ball ----------
    if ball is not None:
        tid = int(ball["track_id"])
        x1, y1, x2, y2 = map(int, ball["bbox"])

        BALL_COLOR = (0, 255, 255)  # yellow
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), BALL_COLOR, 2)
        cv2.putText(
            frame_bgr,
            f"BALL {tid}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BALL_COLOR,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video-path", required=True, help="Path to match video")

    # Keep --conf (sets both), plus allow separate knobs
    parser.add_argument("--conf", type=float, default=None, help="Set BOTH conf-player and conf-ball")
    parser.add_argument("--conf-player", type=float, default=0.35, help="YOLO confidence for players")
    parser.add_argument("--conf-ball", type=float, default=0.05, help="YOLO confidence for ball")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size (bigger helps ball)")

    parser.add_argument("--trail-len", type=int, default=25, help="Trail length in points")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")

    parser.add_argument("--save-video", default="", help="Optional output video path (mp4)")

    parser.add_argument(
        "--pixels-to-meters",
        type=float,
        default=0.01,
        help="Rough conversion for analytics (temporary). Example: 0.01 means 1px ≈ 1cm",
    )

    # Day 16: heatmap visualization / saving
    parser.add_argument("--show-heatmap", action="store_true", help="Overlay heatmap live on video")
    parser.add_argument("--heatmap-alpha", type=float, default=0.35, help="Heatmap overlay strength (0..1)")
    parser.add_argument("--heatmap-blur", type=int, default=0, help="Optional blur kernel size (odd number)")
    parser.add_argument("--save-heatmap", default="", help="Save heatmap PNG at end (e.g., ../docs/images/day16_heatmap.png)")
    parser.add_argument("--save-heatmap-npy", default="", help="Save raw heatmap grid npy at end")

    args = parser.parse_args()

    # if --conf is provided, override both
    conf_player = args.conf_player
    conf_ball = args.conf_ball
    if args.conf is not None:
        conf_player = float(args.conf)
        conf_ball = float(args.conf)

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    fps = float(fps)

    pipeline = VisionPipeline(
        model_path="yolov8n.pt",
        conf_player=conf_player,
        conf_ball=conf_ball,
        imgsz=args.imgsz,
    )

    match_state = MatchState(
        fps=fps,
        pixels_to_meters=args.pixels_to_meters,
        history_len=200,
        video_width=video_w,
        video_height=video_h,
        heatmap_grid_x=60,
        heatmap_grid_y=40,
    )

    writer = None
    save_path: Optional[Path] = None
    if args.save_video:
        save_path = Path(args.save_video).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (video_w, video_h))
        if not writer.isOpened():
            writer = None
            print("⚠️ Could not open VideoWriter. Continuing without saving.")

    trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    frame_i = 0
    print(f"Video: {video_path.name} | FPS={fps:.2f} | Size={video_w}x{video_h}")
    print(f"conf_player={conf_player} conf_ball={conf_ball} imgsz={args.imgsz}")
    print("Press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_data = pipeline.process_frame(frame)
        players = frame_data.get("players", [])
        ball = frame_data.get("ball", None)

        match_state.update(players)

        # optional heatmap overlay
        display = frame
        if args.show_heatmap and match_state.heatmap is not None:
            display = match_state.heatmap.overlay_on_frame(
                display,
                alpha=args.heatmap_alpha,
                blur_ksize=args.heatmap_blur,
            )

        draw_tracks_with_stats(
            display,
            players=players,
            ball=ball,
            match_state=match_state,
            trail_history=trail_history,
            trail_len=args.trail_len,
        )

        cv2.imshow("FotStrike AR - Vision Demo (Players + Ball + Stats + Heatmap)", display)

        if writer is not None:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_i += 1
        if args.max_frames and frame_i >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ Saved video: {save_path}")

    # Save heatmap PNG at end
    if args.save_heatmap and match_state.heatmap is not None:
        out_png = match_state.heatmap.save_png(
            args.save_heatmap,
            blur_ksize=args.heatmap_blur,
        )
        print(f"✅ Saved heatmap: {out_png}")

    # Save raw grid npy at end (optional)
    if args.save_heatmap_npy and match_state.heatmap is not None:
        out_npy = Path(args.save_heatmap_npy).expanduser().resolve()
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_npy), match_state.heatmap.grid)
        print(f"✅ Saved heatmap grid: {out_npy}")

    if match_state.heatmap is not None:
        print("Heatmap max cell count:", match_state.heatmap.max_value())

    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
