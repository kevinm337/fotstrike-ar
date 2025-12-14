from pathlib import Path
from typing import Optional
import cv2


def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 2.0,
    max_frames: Optional[int] = None,
    prefix: str = "frame",
) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps != video_fps:
        video_fps = 30.0

    step = max(1, int(round(video_fps / target_fps)))

    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            out_path = out_dir / f"{prefix}_{saved:06d}.png"
            ok_write = cv2.imwrite(str(out_path), frame)
            if not ok_write:
                raise RuntimeError(f"Failed to write frame: {out_path}")

            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved
