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
    y = (track_id * 67) % 255
    z = (track_id * 97) % 255
    return int(x), int(y), int(z)


def bbox_center_xyxy(bbox_xyxy: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return cx, cy


def _ensure_odd_ksize(k: int) -> int:
    if k <= 0:
        return 0
    return k if (k % 2 == 1) else (k + 1)


def _blend_heatmap(frame_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend a heatmap image onto frame using alpha in [0,1].
    """
    a = float(alpha)
    if a <= 0:
        return frame_bgr
    if a >= 1:
        return heat_bgr
    return cv2.addWeighted(frame_bgr, 1.0 - a, heat_bgr, a, 0.0)


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
            speed = float(getattr(st, "current_speed", 0.0))
            dist = float(getattr(st, "total_distance", 0.0))
            text = f"ID {tid} | {speed:.2f} m/s | {dist:.1f} m"
        else:
            text = f"ID {tid}"

        # pressure icon (Day 17)
        if st is not None and getattr(st, "under_pressure", False):
            icon_x = min(x2 + 10, frame_bgr.shape[1] - 1)
            icon_y = min(max(y1 + 10, 0), frame_bgr.shape[0] - 1)
            cv2.circle(frame_bgr, (int(icon_x), int(icon_y)), 8, (0, 0, 255), 2)
            cv2.putText(
                frame_bgr,
                "!",
                (int(icon_x) - 4, int(icon_y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

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

    # Support BOTH flags:
    # - old style: --video-path (your command)
    # - new/alt:  --video
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--video", type=str, default="")

    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--conf-player", type=float, default=0.35)
    parser.add_argument("--conf-ball", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--trail-len", type=int, default=25)
    parser.add_argument("--pixels-to-meters", type=float, default=0.01)

    # Day 17 pressure controls
    parser.add_argument("--pressure-radius-px", type=float, default=60.0)
    parser.add_argument("--pressure-threshold", type=int, default=2)

    # Heatmap (Day 16) — matches your command flags
    parser.add_argument("--show-heatmap", action="store_true")
    parser.add_argument("--heatmap-alpha", type=float, default=0.35)
    parser.add_argument("--heatmap-blur", type=int, default=21)
    parser.add_argument("--save-heatmap", type=str, default="")

    args = parser.parse_args()

    # Resolve which video path was provided
    video_arg = args.video_path or args.video
    if not video_arg:
        raise SystemExit("Error: you must pass --video-path <path> or --video <path>")

    video_path = Path(video_arg).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fps = float(fps)

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = VisionPipeline(
        model_path="yolov8n.pt",
        conf_player=args.conf_player,
        conf_ball=args.conf_ball,
        imgsz=args.imgsz,
    )

    match_state = MatchState(
        fps=fps,
        pixels_to_meters=args.pixels_to_meters,
        pressure_radius_px=args.pressure_radius_px,
        pressure_threshold=args.pressure_threshold,
        history_len=200,
        video_width=video_w,
        video_height=video_h,
    )

    out_writer = None
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (video_w, video_h))

    trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    frame_i = 0
    processed = 0

    blur_k = _ensure_odd_ksize(int(args.heatmap_blur))
    alpha = float(args.heatmap_alpha)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_i < args.start_frame:
            frame_i += 1
            continue

        if (frame_i - args.start_frame) % args.stride != 0:
            frame_i += 1
            continue

        out = pipeline.process_frame(frame_bgr)
        players = out.get("players", [])
        ball = out.get("ball", None)

        # updates distance/speed + pressure flags inside MatchState
        match_state.update(players)

        # Optional: heatmap overlay
        if args.show_heatmap and getattr(match_state, "heatmap", None) is not None:
            heat_img = match_state.heatmap.render_colormap(blur_ksize=blur_k)
            # heat_img is BGR image same size as frame
            frame_bgr = _blend_heatmap(frame_bgr, heat_img, alpha)

        draw_tracks_with_stats(
            frame_bgr=frame_bgr,
            players=players,
            ball=ball,
            match_state=match_state,
            trail_history=trail_history,
            trail_len=args.trail_len,
        )

        cv2.imshow("FotStrike AR - Vision Demo", frame_bgr)
        if out_writer is not None:
            out_writer.write(frame_bgr)

        processed += 1
        frame_i += 1

        if processed >= args.max_frames:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    # Save heatmap image at end (Day 16)
    if args.save_heatmap and getattr(match_state, "heatmap", None) is not None:
        save_path = Path(args.save_heatmap).expanduser().resolve()
        match_state.heatmap.save_png(save_path, blur_ksize=blur_k)
        print(f"[heatmap] saved to: {save_path}")


if __name__ == "__main__":
    main()
