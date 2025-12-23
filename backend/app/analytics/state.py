from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


def center_from_bbox_xyxy(bbox: List[float]) -> Tuple[float, float]:
    """
    bbox: [x1, y1, x2, y2] in pixels
    returns: (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PlayerState(BaseModel):
    track_id: int

    # last known info
    last_position: Optional[Tuple[float, float]] = None
    last_timestamp: Optional[float] = None  # seconds since start

    # analytics
    total_distance: float = 0.0            # pixels (for now)
    current_speed: float = 0.0             # pixels/sec (for now)

    # history (used later for heatmaps, sprints, etc.)
    positions: List[Tuple[float, float]] = Field(default_factory=list)


class MatchState:
    """
    Holds rolling analytics for a match.

    NOTE: Right now "distance" and "speed" are in pixel units because we don't yet
    map pixels -> meters. Later we will calibrate using pitch lines / homography.
    """

    def __init__(self, fps: float):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = float(fps)
        self.players: Dict[int, PlayerState] = {}
        self.frame_index: int = 0

    def _timestamp(self) -> float:
        """Seconds since start for current frame."""
        return self.frame_index / self.fps

    def update(self, tracked_players: List[dict]) -> None:
        """
        tracked_players item format (from VisionPipeline):
          {
            "track_id": int,
            "bbox": [x1,y1,x2,y2],
            "label": "player"
          }

        On each frame:
          - compute center position
          - compute delta distance from last frame (per track_id)
          - compute speed = distance / dt
          - accumulate total distance
          - store positions history
        """
        t_now = self._timestamp()
        dt = 1.0 / self.fps  # fixed timestep per processed frame

        for p in tracked_players:
            if p.get("label") != "player":
                continue

            track_id = int(p["track_id"])
            bbox = p["bbox"]
            cx, cy = center_from_bbox_xyxy(bbox)
            pos = (float(cx), float(cy))

            if track_id not in self.players:
                # first time we see this player ID
                self.players[track_id] = PlayerState(
                    track_id=track_id,
                    last_position=pos,
                    last_timestamp=t_now,
                    total_distance=0.0,
                    current_speed=0.0,
                    positions=[pos],
                )
                continue

            st = self.players[track_id]

            if st.last_position is None:
                st.last_position = pos
                st.last_timestamp = t_now
                st.positions.append(pos)
                continue

            # distance between last position and current
            d = euclidean(pos, st.last_position)

            # speed (pixels/sec). Later convert to m/s after calibration.
            speed = d / dt if dt > 0 else 0.0

            st.total_distance += d
            st.current_speed = speed
            st.last_position = pos
            st.last_timestamp = t_now
            st.positions.append(pos)

        # advance frame counter AFTER processing
        self.frame_index += 1
