from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from app.vision.pipeline import VisionPipeline
from app.analytics.state import MatchState
from app.schemas.frame import FrameOut, PlayerOut, BallOut

app = FastAPI(title="FotStrike AR API", version="0.3.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _serialize_player(p: dict, match_state: MatchState) -> Dict[str, Any]:
    """Convert internal player dict to the WS JSON shape."""
    tid = int(p.get("track_id", -1))
    x1, y1, x2, y2 = [float(v) for v in p.get("bbox", [0, 0, 0, 0])]
    st = match_state.players.get(tid)

    return {
        "id": tid,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "speed": float(getattr(st, "current_speed", 0.0)) if st is not None else 0.0,
        "under_pressure": bool(getattr(st, "under_pressure", False)) if st is not None else False,
        "opponents_in_radius": int(getattr(st, "opponents_in_radius", 0)) if st is not None else 0,
    }


def _serialize_ball(ball: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not ball:
        return None
    tid = int(ball.get("track_id", -1))
    x1, y1, x2, y2 = [float(v) for v in ball.get("bbox", [0, 0, 0, 0])]
    return {"id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Stream real pipeline output over WebSocket.

    Query params (optional):
      - video_path: str  (default: ../data/matches/match1.mp4)
      - max_frames: int  (default: 300; -1 for unlimited)
      - stride: int      (default: 1)
      - conf_player: float (default: 0.35)
      - conf_ball: float   (default: 0.05)
      - imgsz: int         (default: 960)
      - pixels_to_meters: float (default: 0.01)
      - pressure_radius_px: float (default: 60.0)
      - pressure_threshold: int   (default: 2)
      - loop: int (0/1) (default: 0)  # loop video when it ends
      - send_every_ms: int (default: 0)  # throttle sending; 0 => use 1/fps
    """
    await ws.accept()

    qp = ws.query_params
    video_path = qp.get("video_path", "../data/matches/match1.mp4")
    max_frames = int(qp.get("max_frames", "300"))
    stride = max(1, int(qp.get("stride", "1")))
    conf_player = float(qp.get("conf_player", "0.35"))
    conf_ball = float(qp.get("conf_ball", "0.05"))
    imgsz = int(qp.get("imgsz", "960"))
    pixels_to_meters = float(qp.get("pixels_to_meters", "0.01"))
    pressure_radius_px = float(qp.get("pressure_radius_px", "60.0"))
    pressure_threshold = int(qp.get("pressure_threshold", "2"))
    loop = int(qp.get("loop", "0")) == 1
    send_every_ms = int(qp.get("send_every_ms", "0"))

    # Resolve video path relative to current working dir (backend/)
    vp = Path(video_path).expanduser()
    if not vp.is_absolute():
        vp = (Path.cwd() / vp).resolve()

    if not vp.exists():
        await ws.send_json({"error": f"video not found: {str(vp)}"})
        await ws.close()
        return

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        await ws.send_json({"error": f"could not open video: {str(vp)}"})
        await ws.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fps = float(fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = VisionPipeline(
        model_path="yolov8n.pt",
        conf_player=conf_player,
        conf_ball=conf_ball,
        imgsz=imgsz,
    )

    match_state = MatchState(
        fps=fps,
        pixels_to_meters=pixels_to_meters,
        pressure_radius_px=pressure_radius_px,
        pressure_threshold=pressure_threshold,
        history_len=200,
        video_width=w,
        video_height=h,
    )

    # throttle interval
    if send_every_ms > 0:
        sleep_s = send_every_ms / 1000.0
    else:
        sleep_s = 1.0 / fps if fps > 0 else 0.04

    frame_id = 0
    read_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                await ws.send_json({"event": "eof", "frame_id": frame_id})
                await ws.close()
                break

            # stride (skip frames)
            if read_i % stride != 0:
                read_i += 1
                continue
            read_i += 1

            out = pipeline.process_frame(frame)
            players_raw: List[dict] = out.get("players", [])
            ball_raw: Optional[dict] = out.get("ball", None)

            # Update analytics state (speed, distance, under_pressure)
            match_state.update(players_raw)

            players = [
                _serialize_player(p, match_state)
                for p in players_raw
                if p.get("label") == "player"
            ]

            payload = {
                "frame_id": frame_id,
                "players": players,
                "ball": _serialize_ball(ball_raw),
            }

            await ws.send_json(payload)
            frame_id += 1

            if max_frames != -1 and frame_id >= max_frames:
                await ws.send_json({"event": "max_frames_reached", "frame_id": frame_id})
                await ws.close()
                break

            # Try to approximate real-time streaming
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    except WebSocketDisconnect:
        # client closed the connection
        pass
    except Exception as e:
        # send error once if possible
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    # Run:
    #   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
    # or:
    #   python -m app.main
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
